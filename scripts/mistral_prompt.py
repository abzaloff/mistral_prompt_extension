# extensions/<your_extension>/scripts/mistral_prompt.py
# Minimal UI + clipboard button only:
# - Drag&drop / click / "Paste from clipboard" button
# - Remove last / Clear all / Individual delete buttons
# - Presets + inline editor
# - Send to a selected vision API + insert into main prompt

import io
import json
import base64
import requests
from urllib.parse import quote, urlsplit, urlunsplit
from PIL import Image

import gradio as gr
from modules import scripts, shared, processing, sd_models, timer
from modules.ui_components import ToolButton

MAX_IMAGES = 30
API_URL = "https://api.mistral.ai/v1/chat/completions"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_VISION_MODEL = "qwen/qwen3.6-27b"
GROQ_MAX_IMAGES = 3
LMSTUDIO_DEFAULT_API_BASE = "http://127.0.0.1:1234/v1"
GEMINI_FREE_TIER_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
GEMINI_MODELS_WITHOUT_SAMPLING = {
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
}
MODEL_CHOICES = [
    "mistral: mistral-large-2512",
    "mistral: mistral-medium-latest",
    "mistral: mistral-small-latest",
    "mistral: ministral-14b-latest",
    f"groq: {GROQ_VISION_MODEL}",
    *[f"gemini: {model}" for model in GEMINI_FREE_TIER_MODELS],
]
DEFAULT_MODEL_CHOICE = MODEL_CHOICES[0]

# ========= Presets =========
PRESETS_OPT_KEY = "mistral_presets_json"
OLD_LMSTUDIO_LLM_PRESET = "Improve the prompt. Analyze the tokens and rewrite the prompt in clear English. Give an expanded answer in English."
LMSTUDIO_LLM_PRESET = "Rewrite the user's input into one clear, detailed English image prompt. Accept any input format, including comma-separated tokens, short notes, fragments, or natural language. Expand the idea into a coherent descriptive prompt with richer visual detail, while preserving the original subject, mood, style, composition, and important attributes. Do not add unrelated elements. Output only one final prompt in English. Do not include explanations, labels, introductions, alternatives, bullet points, quotation marks, or any extra text."
DEFAULT_PRESETS = {
    "Flux - Describe": "Describe the image",
    "SDXL - Tokens": "Describe the image using only comma-separated tokens",
    "LM Studio LLM": LMSTUDIO_LLM_PRESET,
}

def _ensure_presets_in_opts():
    raw = shared.opts.data.get(PRESETS_OPT_KEY, "").strip()
    if not raw:
        set_presets(DEFAULT_PRESETS, merge_defaults=False)
        return

    try:
        data = json.loads(raw)
    except Exception:
        return

    if not isinstance(data, dict):
        return

    changed = False
    if data.get("LM Studio LLM") == OLD_LMSTUDIO_LLM_PRESET:
        data["LM Studio LLM"] = LMSTUDIO_LLM_PRESET
        changed = True
    if changed:
        set_presets(data, merge_defaults=False)

def get_presets():
    _ensure_presets_in_opts()
    raw = shared.opts.data.get(PRESETS_OPT_KEY, "{}")
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else dict(DEFAULT_PRESETS)
    except Exception:
        return dict(DEFAULT_PRESETS)

def set_presets(presets: dict, merge_defaults: bool = False):
    data = dict(presets or {})
    if merge_defaults:
        for name, text in DEFAULT_PRESETS.items():
            data.setdefault(name, text)
    value = json.dumps(data, ensure_ascii=False)
    try:
        if PRESETS_OPT_KEY in shared.opts.data_labels:
            shared.opts.set(PRESETS_OPT_KEY, value, run_callbacks=False)
        else:
            shared.opts.data[PRESETS_OPT_KEY] = value
        shared.opts.save(shared.config_filename)
    except Exception as exc:
        try:
            shared.opts.data[PRESETS_OPT_KEY] = value
            shared.opts.save(shared.config_filename)
        except Exception as fallback_exc:
            raise RuntimeError(f"Failed to save Mistral presets to config: {fallback_exc}") from fallback_exc

# ========= API backends =========

# Reuse one HTTP session to persist cookies (helps with Cloudflare checks).
_mistral_session = None
_gemini_session = None
_groq_session = None
_lmstudio_session = None

def get_mistral_session():
    global _mistral_session
    if _mistral_session is None:
        _mistral_session = requests.Session()
    return _mistral_session

def get_gemini_session():
    global _gemini_session
    if _gemini_session is None:
        _gemini_session = requests.Session()
    return _gemini_session

def get_groq_session():
    global _groq_session
    if _groq_session is None:
        _groq_session = requests.Session()
    return _groq_session

def get_lmstudio_session():
    global _lmstudio_session
    if _lmstudio_session is None:
        _lmstudio_session = requests.Session()
    return _lmstudio_session

def get_lmstudio_api_base():
    return (shared.opts.data.get("lmstudio_api_base", LMSTUDIO_DEFAULT_API_BASE) or LMSTUDIO_DEFAULT_API_BASE).strip().rstrip("/")

def get_lmstudio_native_api_base():
    api_base = get_lmstudio_api_base()
    parsed = urlsplit(api_base)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3].rstrip("/")
    path = f"{path}/api/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))

def get_lmstudio_headers():
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    api_key = (shared.opts.data.get("lmstudio_api_key", "") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers

def fetch_lmstudio_model_choices(timeout=5):
    api_base = get_lmstudio_api_base()
    resp = get_lmstudio_session().get(
        f"{api_base}/models",
        headers=get_lmstudio_headers(),
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    models = []
    for item in payload.get("data", []):
        model_id = item.get("id") if isinstance(item, dict) else None
        if model_id:
            models.append(f"lmstudio: {model_id}")
    return sorted(set(models))

def unload_lmstudio_model(model, timeout=10):
    api_base = get_lmstudio_native_api_base()
    resp = get_lmstudio_session().post(
        f"{api_base}/models/unload",
        headers=get_lmstudio_headers(),
        json={"instance_id": model},
        timeout=timeout,
    )
    if not resp.ok:
        try:
            err = resp.json().get("error", {})
            message = err.get("message") if isinstance(err, dict) else None
            message = message or resp.text
        except Exception:
            message = resp.text or resp.reason
        raise ValueError(f"LM Studio unload error {resp.status_code}: {message}")
    return resp.json()

def unload_forge_models():
    t = timer.Timer()
    sd_models.unload_model_weights()
    t.record("unload all models")
    print(f"Unloaded all models and cleared RAM/VRAM in {t.total:.1f}s")

def get_model_choices(include_lmstudio=True, lmstudio_timeout=1.5):
    choices = list(MODEL_CHOICES)
    if include_lmstudio:
        try:
            choices.extend(fetch_lmstudio_model_choices(timeout=lmstudio_timeout))
        except Exception:
            pass
    return choices

def encode_image_for_request(img):
    # Read image constraints from extension settings.
    max_size = int(shared.opts.data.get("mistral_image_max_size", 768))
    max_kb = int(shared.opts.data.get("mistral_image_max_kb", 400))

    # Downscale large images before upload.
    if img.width > max_size or img.height > max_size:
        ratio = min(max_size / img.width, max_size / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Shrink JPEG quality until target size is reached.
    quality = 90
    buf = io.BytesIO()

    while True:
        buf.seek(0)
        buf.truncate()

        img.save(buf, format="JPEG", quality=quality)
        size_kb = buf.tell() / 1024

        if size_kb <= max_kb or quality <= 40:
            break

        quality -= 5

    return base64.b64encode(buf.getvalue()).decode("utf-8")

def normalize_model_choice(model_choice):
    model_choice = (model_choice or DEFAULT_MODEL_CHOICE).strip()
    if ":" not in model_choice:
        model_choice = DEFAULT_MODEL_CHOICE

    provider, model = model_choice.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if not model or provider not in ("mistral", "gemini", "groq", "lmstudio"):
        return normalize_model_choice(DEFAULT_MODEL_CHOICE)
    return provider, model

def build_prompt_for_request(instruction_prompt, source_prompt):
    instruction_prompt = (instruction_prompt or "").strip()
    source_prompt = (source_prompt or "").strip()
    if not source_prompt:
        return instruction_prompt
    if not instruction_prompt:
        return source_prompt
    return f"{instruction_prompt}\n\nPrompt to improve:\n{source_prompt}"

def send_to_mistral(model, prompt, images, temperature, maximum_tokens, top_p):
    api_key = shared.opts.data.get("mistral_api_key", "").strip()
    if not api_key:
        raise ValueError("Mistral API key is not set in Settings.")

    image_urls = []
    for img in images:
        b64 = encode_image_for_request(img)
        image_urls.append(f"data:image/jpeg;base64,{b64}")

    if len(image_urls) > MAX_IMAGES:
        raise ValueError(f"Maximum {MAX_IMAGES} images supported.")

    content_list = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content_list.append({"type": "image_url", "image_url": url})

    data = {
        "model": model,
        "messages": [{"role": "user", "content": content_list}],
        "temperature": float(temperature),
        "max_tokens": int(maximum_tokens),
        "top_p": float(top_p),
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    session = get_mistral_session()
    resp = session.post(API_URL, headers=headers, json=data, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def send_to_gemini(model, prompt, images, temperature, maximum_tokens, top_p):
    api_key = shared.opts.data.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Gemini API key is not set in Settings.")

    if not (prompt or "").strip():
        raise ValueError("Prompt is empty.")

    if len(images or []) > MAX_IMAGES:
        raise ValueError(f"Maximum {MAX_IMAGES} images supported.")

    parts = []
    for img in images or []:
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": encode_image_for_request(img),
            }
        })
    parts.append({"text": prompt})

    generation_config = {
        "maxOutputTokens": int(maximum_tokens),
    }
    if model not in GEMINI_MODELS_WITHOUT_SAMPLING:
        generation_config.update({
            "temperature": float(temperature),
            "topP": float(top_p),
        })

    data = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "x-goog-api-key": api_key,
    }

    url = f"{GEMINI_API_BASE}/{quote(model, safe='')}:generateContent"
    resp = get_gemini_session().post(url, headers=headers, json=data, timeout=120)
    if not resp.ok:
        try:
            err = resp.json().get("error", {})
            message = err.get("message") or resp.text
            status = err.get("status")
            if status:
                message = f"{status}: {message}"
        except Exception:
            message = resp.text or resp.reason
        if resp.status_code in (401, 403):
            raise ValueError(f"Gemini authorization failed: {message}")
        if resp.status_code == 429:
            raise ValueError(f"Gemini rate limit exceeded: {message}")
        raise ValueError(f"Gemini API error {resp.status_code}: {message}")

    payload = resp.json()
    candidates = payload.get("candidates") or []
    if not candidates:
        prompt_feedback = payload.get("promptFeedback") or {}
        block_reason = prompt_feedback.get("blockReason")
        if block_reason:
            raise ValueError(f"Gemini blocked the prompt: {block_reason}")
        raise ValueError("Gemini returned no candidates.")

    content = candidates[0].get("content") or {}
    text = "\n".join(part.get("text", "") for part in content.get("parts", []) if part.get("text"))
    if text.strip():
        return text.strip()

    finish_reason = candidates[0].get("finishReason")
    if finish_reason:
        raise ValueError(f"Gemini returned no text. Finish reason: {finish_reason}")
    raise ValueError("Gemini returned an empty response.")

def send_to_groq(model, prompt, images, temperature, maximum_tokens, top_p):
    api_key = (shared.opts.data.get("groq_api_key", "") or "").strip()
    if not api_key:
        raise ValueError("Groq API key is not set in Settings.")

    if not (prompt or "").strip():
        raise ValueError("Prompt is empty.")

    if model != GROQ_VISION_MODEL:
        raise ValueError(f"Unsupported Groq model: {model}")

    if len(images or []) > GROQ_MAX_IMAGES:
        raise ValueError(
            f"Groq {GROQ_VISION_MODEL} supports a maximum of "
            f"{GROQ_MAX_IMAGES} images per request."
        )

    content_list = [{"type": "text", "text": prompt}]
    for img in images or []:
        b64 = encode_image_for_request(img)
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    data = {
        "model": model,
        "messages": [{"role": "user", "content": content_list}],
        "temperature": float(temperature),
        "max_completion_tokens": int(maximum_tokens),
        "top_p": float(top_p),
        "reasoning_effort": "none",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    resp = get_groq_session().post(
        GROQ_API_URL,
        headers=headers,
        json=data,
        timeout=120,
    )
    if not resp.ok:
        try:
            err = resp.json().get("error", {})
            message = err.get("message") if isinstance(err, dict) else None
            message = message or resp.text
        except Exception:
            message = resp.text or resp.reason
        if resp.status_code in (401, 403):
            raise ValueError(f"Groq authorization failed: {message}")
        if resp.status_code == 429:
            raise ValueError(f"Groq rate limit exceeded: {message}")
        raise ValueError(f"Groq API error {resp.status_code}: {message}")

    payload = resp.json()
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("Groq returned no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        text = "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("text")
        )
    else:
        text = str(content or "")
    if text.strip():
        return text.strip()
    raise ValueError("Groq returned an empty response.")

def send_to_lmstudio(model, prompt, images, temperature, maximum_tokens, top_p):
    if not (prompt or "").strip():
        raise ValueError("Prompt is empty.")

    if len(images or []) > MAX_IMAGES:
        raise ValueError(f"Maximum {MAX_IMAGES} images supported.")

    content_list = [{"type": "text", "text": prompt}]
    for img in images or []:
        b64 = encode_image_for_request(img)
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    data = {
        "model": model,
        "messages": [{"role": "user", "content": content_list}],
        "temperature": float(temperature),
        "max_tokens": int(maximum_tokens),
        "top_p": float(top_p),
    }

    api_base = get_lmstudio_api_base()
    resp = get_lmstudio_session().post(
        f"{api_base}/chat/completions",
        headers=get_lmstudio_headers(),
        json=data,
        timeout=120,
    )
    if not resp.ok:
        try:
            err = resp.json().get("error", {})
            message = err.get("message") if isinstance(err, dict) else None
            message = message or resp.text
        except Exception:
            message = resp.text or resp.reason
        raise ValueError(f"LM Studio API error {resp.status_code}: {message}")

    payload = resp.json()
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("LM Studio returned no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        text = "\n".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("text"))
    else:
        text = str(content or "")
    if text.strip():
        return text.strip()
    raise ValueError("LM Studio returned an empty response.")

def send_to_selected_model(model_choice, prompt, images, temperature, maximum_tokens, top_p):
    provider, model = normalize_model_choice(model_choice)
    if provider == "lmstudio":
        return send_to_lmstudio(model, prompt, images, temperature, maximum_tokens, top_p)
    if provider == "gemini":
        return send_to_gemini(model, prompt, images, temperature, maximum_tokens, top_p)
    if provider == "groq":
        return send_to_groq(model, prompt, images, temperature, maximum_tokens, top_p)
    return send_to_mistral(model, prompt, images, temperature, maximum_tokens, top_p)

# ========= UI Script =========

class Script(scripts.Script):
    def title(self):
        return "Mistral++"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        _ensure_presets_in_opts()

        with gr.Accordion(
            "Mistral++",
            open=False,
            elem_id=self.elem_id("mp_root"),
            elem_classes=["mp-extension-root"],
        ):

            # ===== CSS + JS: dropzone visuals + delete buttons =====
            gr.HTML(
                """
<style>
  :root{
    --mp-gap:8px;
    --mp-gap-tight:6px;
    --mp-radius:8px;
    --mp-control-height:42px;
    --mp-thumb-size:120px;
    --mp-delete-size:24px;
  }
  .mp-compact-row{
    gap:var(--mp-gap) !important;
    align-items:flex-end !important;
  }
  .mp-action-grid{
    display:grid !important;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:var(--mp-gap);
    align-items:stretch;
  }
  .mp-two-column-grid{
    display:grid !important;
    grid-template-columns:repeat(2,minmax(0,1fr));
    gap:var(--mp-gap);
    align-items:stretch;
  }
  .mp-action-grid .gr-button,
  .mp-two-column-grid .gr-button,
  .mp-compact-row .gr-button{
    width:100% !important;
  }
  .mp-action-grid .gr-button,
  .mp-two-column-grid .gr-button,
  .mp-action-grid button,
  .mp-two-column-grid button,
  .mp-refresh-lmstudio-models,
  .mp-refresh-lmstudio-models.gr-button,
  .mp-refresh-lmstudio-models button{
    min-height:var(--mp-control-height) !important;
  }
  .mp-rounded-btn,
  .mp-rounded-btn.gr-button,
  .mp-rounded-btn button{
    border-radius:var(--mp-radius) !important;
  }

  /* model and preset controls share the same responsive two-column layout */
  .mp-primary-control-row{
    gap:var(--mp-gap) !important;
    align-items:flex-end !important;
  }
  .mp-primary-control-row > *{
    min-width:0;
  }
  .mp-preset-bar{align-items:flex-end !important;}
  .mp-preset-bar label{display:none !important;}
  .mp-preset-bar .gr-form{margin-bottom:0 !important;}
  .mp-preset-bar .gr-button{white-space:nowrap;width:100% !important;}
  .mp-preset-actions{
    width:100% !important;
    gap:var(--mp-gap) !important;
  }
  .mp-system-prompt-header{
    display:flex !important;
    flex-wrap:wrap !important;
    gap:18px !important;
    align-items:center !important;
    margin-top:var(--mp-gap-tight) !important;
    margin-bottom:4px !important;
  }
  .mp-system-prompt-header > *{
    flex:0 0 auto !important;
    width:auto !important;
    min-width:0 !important;
  }
  .mp-system-prompt-title{
    flex:0 0 auto !important;
    width:auto !important;
    min-width:0 !important;
    margin:0 !important;
    padding:0 !important;
    border:none !important;
    background:transparent !important;
  }
  .mp-system-prompt-title .wrap{
    display:flex !important;
    align-items:center !important;
    height:auto !important;
  }
  .mp-system-prompt-title span{
    color:var(--block-title-text-color) !important;
    font-size:var(--block-title-text-size) !important;
    font-weight:var(--block-title-text-weight) !important;
    line-height:var(--line-md) !important;
    position:relative;
    top:-1px;
  }
  .mp-system-prompt-header .mp-improve-prompt-enabled{
    flex:0 1 auto !important;
    width:auto !important;
    max-width:100% !important;
    min-width:0 !important;
    min-height:24px !important;
    margin:0 !important;
  }
  .mp-system-prompt-header .mp-improve-prompt-enabled label{
    width:100% !important;
    max-width:100% !important;
    margin:0 !important;
    white-space:normal !important;
    overflow-wrap:anywhere !important;
  }
  .mp-system-prompt-header .mp-improve-prompt-enabled label > span{
    flex:1 1 0 !important;
    max-width:100% !important;
    min-width:0 !important;
    white-space:normal !important;
    overflow-wrap:anywhere !important;
    word-break:break-word !important;
  }
  .mp-system-prompt-header .mp-improve-prompt-enabled input{
    flex:0 0 auto;
  }
  @media (max-width:640px){
    .mp-system-prompt-header{
      display:grid !important;
      grid-template-columns:minmax(0,1fr) !important;
      column-gap:0 !important;
      row-gap:4px !important;
    }
    .mp-system-prompt-header > *{
      width:100% !important;
      max-width:100% !important;
      min-width:0 !important;
    }
    .mp-system-prompt-header .mp-improve-prompt-enabled{
      width:100% !important;
      max-width:100% !important;
      min-width:0 !important;
    }
    .mp-system-prompt-header .mp-improve-prompt-enabled label{
      width:100% !important;
      max-width:100% !important;
      min-width:0 !important;
    }
  }
  .mp-system-prompt-row{
    gap:var(--mp-gap) !important;
    align-items:stretch !important;
  }
  .mp-system-prompt textarea{
    min-height:120px !important;
    max-height:120px !important;
    overflow-y:auto !important;
    resize:vertical !important;
  }
  .mp-model-output textarea{
    max-height:120px !important;
    overflow-y:auto !important;
  }
  .mp-preset-modal{
    position:fixed !important;
    inset:0 !important;
    z-index:2147483000 !important;
    align-items:center !important;
    justify-content:center !important;
    padding:24px !important;
    background:rgba(0,0,0,.56) !important;
    border:none !important;
    box-shadow:none !important;
    contain:none !important;
    pointer-events:auto !important;
    user-select:none !important;
  }
  #txt2img_settings:has(.mp-preset-modal:not(.hide)),
  #txt2img_results:has(.mp-preset-modal:not(.hide)),
  #img2img_settings:has(.mp-preset-modal:not(.hide)),
  #img2img_results:has(.mp-preset-modal:not(.hide)){
    z-index:2147482999 !important;
  }
  .mp-preset-modal[style*="display: block"]{display:flex !important;}
  .mp-preset-modal > .wrap,
  .mp-preset-modal > div{
    position:relative !important;
    z-index:2147483001 !important;
    width:min(760px, calc(100vw - 48px)) !important;
  }
  .mp-preset-modal .block,
  .mp-preset-modal .gr-box{
    border-color:var(--block-border-color) !important;
  }
  .mp-preset-modal-panel{
    position:relative !important;
    z-index:2147483002 !important;
    width:min(760px, calc(100vw - 48px)) !important;
    max-height:calc(100vh - 48px) !important;
    overflow:auto !important;
    padding:16px !important;
    border:1px solid var(--block-border-color) !important;
    border-radius:var(--mp-radius) !important;
    background:#0b1118 !important;
    box-shadow:0 18px 55px rgba(0,0,0,.45) !important;
    isolation:isolate !important;
  }
  .mp-preset-modal-panel::before{
    content:"" !important;
    position:absolute !important;
    inset:0 !important;
    z-index:0 !important;
    border-radius:var(--mp-radius) !important;
    background:#0b1118 !important;
    pointer-events:none !important;
  }
  .mp-preset-modal-panel .gr-form,
  .mp-preset-modal-panel .wrap,
  .mp-preset-modal-panel input,
  .mp-preset-modal-panel textarea,
  .mp-preset-modal-panel select{
    background:#1f2b38 !important;
  }
  .mp-preset-modal-panel label,
  .mp-preset-modal-panel .block,
  .mp-preset-modal-panel .gr-box{
    background:#0b1118 !important;
  }
  .mp-preset-modal-panel > .styler{
    position:relative !important;
    z-index:1 !important;
  }
  .mp-preset-modal-panel textarea,
  .mp-preset-modal-panel input{
    user-select:text !important;
  }
  .mp-preset-modal-panel h3{margin-top:0 !important;}
  .mp-preset-modal-actions{
    display:flex !important;
    flex-wrap:wrap !important;
    gap:var(--mp-gap) !important;
    align-items:center !important;
  }
  .mp-preset-modal-actions .gr-button,
  .mp-preset-modal-actions button{
    width:auto !important;
    min-width:72px !important;
  }
  .mp-preset-editor-text textarea{
    min-height:220px !important;
    max-height:360px !important;
    overflow-y:auto !important;
    resize:vertical !important;
  }

  /* upload toolbar: three equal buttons */
  .mp-upload-bar{display:grid !important;grid-template-columns:repeat(3,minmax(0,1fr));gap:var(--mp-gap);align-items:stretch;margin-top:var(--mp-gap-tight) !important;}
  .mp-upload-bar .gr-button{width:100%}

  /* output actions */
  .mp-output-actions{display:grid !important;grid-template-columns:repeat(2,minmax(0,1fr));gap:var(--mp-gap);align-items:stretch;}
  .mp-output-actions .gr-button{width:100% !important;min-height:var(--mp-control-height) !important;}

  /* keep LM Studio controls aligned as one compact row */
  .mp-model-bar{align-items:flex-end !important;}
  .mp-model-bar .gr-form{min-width:0 !important;}
  .mp-model-bar .gr-button{width:100% !important;}
  .mp-lmstudio-memory-controls{
    flex-wrap:nowrap !important;
    gap:var(--mp-gap) !important;
    align-items:center !important;
  }
  .mp-vram-cleaner-btn,
  .mp-vram-cleaner-btn.gr-button,
  .mp-vram-cleaner-btn button{
    flex:0 0 var(--mp-control-height) !important;
    width:var(--mp-control-height) !important;
    height:var(--mp-control-height) !important;
    min-width:var(--mp-control-height) !important;
    min-height:var(--mp-control-height) !important;
    padding:0 !important;
    font-size:1rem !important;
    line-height:1 !important;
  }
  .mp-refresh-lmstudio-models,
  .mp-refresh-lmstudio-models.gr-button,
  .mp-refresh-lmstudio-models button{
    height:var(--mp-control-height) !important;
    min-height:var(--mp-control-height) !important;
    padding-top:0 !important;
    padding-bottom:0 !important;
  }
  .mp-lmstudio-auto-unload,
  .mp-lmstudio-auto-unload.gr-checkbox,
  .mp-lmstudio-auto-unload label{
    min-height:var(--mp-control-height) !important;
    align-items:center !important;
  }
  .mp-lmstudio-auto-unload{min-width:190px;}
  /* fixed-height drop zone to avoid layout jumps while uploading */
  .mp-drop{position:relative;isolation:isolate;margin-top:var(--mp-gap) !important;margin-bottom:0;min-height:84px !important;height:84px !important;overflow:hidden;}

  .mp-drop .wrap,
  .mp-drop .file-wrap,
  .mp-drop .border,
  .mp-drop .container{
    height:100% !important;
    min-height:100% !important;
    padding:0 !important;
    background:transparent !important;
    border:none !important;
  }

  /* keep upload status/progress visible */
  .mp-drop [class*="status"],
  .mp-drop [class*="progress"],
  .mp-drop [data-testid*="status"],
  .mp-drop [data-testid*="progress"]{
      position:absolute;
      bottom:6px;
      right:10px;
      z-index:4;
      font-size:12px;
      opacity:.9 !important;
      visibility:visible !important;
  }

  /* hide default Gradio hints in gr.File without breaking click handling */
  .mp-drop label,
  .mp-drop .label,
  .mp-drop .upload-text,
  .mp-drop .filetype,
  .mp-drop p,
  .mp-drop span{
    opacity:0 !important;
  }

  /* custom drop-zone label shown as an overlay */
  .mp-drop::after{
      content:"Drag images here or click to select, or use the \\"Paste from clipboard\\" button";
      position:absolute;
      inset:0;
      display:flex;
      align-items:center;
      justify-content:center;
      padding:0 calc(var(--mp-gap) * 2);
      font-size:13.5px;
      font-weight:600;
      opacity:.95;
      border:1.5px dashed var(--block-border-color);
      border-radius:var(--mp-radius);
      background:var(--body-background-fill);
      text-align:center;
      pointer-events:none;
      z-index:5; /* keep overlay above default Gradio text */
  }
  .mp-drop.dragover::after,
  .mp-drop.border_focus::after{
      content:"Drop to add images";
      border-color:#F87215 !important;
      box-shadow:none !important;
      background:var(--body-background-fill) !important;
  }

  /* gallery with delete buttons */
  .mp-gallery-container{position:relative;margin-top:var(--mp-gap);}
  .mp-custom-gallery .mp-thumbnails{
    display:grid;
    grid-template-columns:repeat(auto-fill,minmax(var(--mp-thumb-size),1fr));
    gap:var(--mp-gap);
  }
  .mp-custom-gallery .thumbnail-item{
    position:relative;
    aspect-ratio:1;
    border-radius:var(--mp-radius);
    overflow:hidden;
  }
  .mp-custom-gallery .thumbnail-item img{
    width:100%;
    height:100%;
    object-fit:cover;
  }

  /* delete button */
  .mp-delete-btn{
    position:absolute;top:4px;right:4px;z-index:10;
    width:var(--mp-delete-size);height:var(--mp-delete-size);border-radius:50%;
    background:rgba(0,0,0,0.75) !important;color:#fff !important;
    border:none;cursor:pointer;
    display:flex;align-items:center;justify-content:center;
    padding:0;
    font-size:15px;line-height:1;
    font-weight:700;
    font-family:Arial,sans-serif;
    transition:background 0.2s;
  }
  .mp-delete-btn:hover{background:rgba(220,38,38,0.9) !important;}
  .mp-delete-pipe-class{display:none !important;}
  
    /* hide empty gallery container by default */
    .mp-gallery-container{
      display:none !important;
      margin:0 !important;
      padding:0 !important;
      border:none !important;
      background:transparent !important;
    }

    /* show container only when at least one image exists */
    .mp-gallery-container:has(img){
      display:block !important;
      margin-top:var(--mp-gap) !important; /* controlled gap when preview appears */
    }
</style>
"""
            )

            with gr.Row(
                elem_id=self.elem_id("mp_model_bar"),
                elem_classes=["mp-compact-row", "mp-primary-control-row", "mp-model-bar"],
            ):
                current_model_choices = get_model_choices()
                initial_is_lmstudio = normalize_model_choice(DEFAULT_MODEL_CHOICE)[0] == "lmstudio"
                with gr.Column(scale=1, min_width=260):
                    model_choice = gr.Dropdown(
                        choices=current_model_choices,
                        value=DEFAULT_MODEL_CHOICE,
                        label="Model",
                        info="Gemini and Groq entries have limited free API quotas that depend on the account and region.",
                    )
                with gr.Column(scale=0, min_width=240, visible=initial_is_lmstudio) as auto_unload_col:
                    with gr.Row(elem_classes=["mp-lmstudio-memory-controls"]):
                        unload_forge_models_btn = ToolButton(
                            value="\U0001f9f9",
                            tooltip="Unload all models and clear RAM/VRAM",
                            elem_id=self.elem_id("mp_unload_all_models"),
                            elem_classes=["mp-vram-cleaner-btn"],
                        )
                        auto_unload_lmstudio = gr.Checkbox(
                            label="Auto-unload LM model",
                            value=False,
                            elem_id=self.elem_id("mp_lmstudio_auto_unload"),
                            elem_classes=["mp-lmstudio-auto-unload"],
                        )
                with gr.Column(scale=1, min_width=260):
                    refresh_lmstudio_models_btn = gr.Button(
                        "Refresh LM Studio models",
                        elem_id=self.elem_id("mp_refresh_lmstudio_models"),
                        elem_classes=["mp-rounded-btn", "mp-refresh-lmstudio-models"],
                    )

            unload_forge_models_btn.click(
                fn=unload_forge_models,
                inputs=[],
                outputs=[],
            )

            def is_lmstudio_choice(model_name):
                provider, _ = normalize_model_choice(model_name)
                return provider == "lmstudio"

            def update_lmstudio_options_visibility(model_name, auto_unload):
                if is_lmstudio_choice(model_name):
                    return (
                        gr.update(visible=True),
                        gr.update(value=True),
                    )
                return (
                    gr.update(visible=False),
                    gr.update(value=False),
                )

            def refresh_lmstudio_models(current_choice, auto_unload):
                try:
                    lmstudio_choices = fetch_lmstudio_model_choices(timeout=5)
                except Exception as exc:
                    choices = list(MODEL_CHOICES)
                    if current_choice and current_choice not in choices:
                        choices.append(current_choice)
                    message = f"Could not refresh LM Studio models: {exc}"
                    if hasattr(gr, "Warning"):
                        gr.Warning(message)
                    return (
                        gr.update(choices=choices, value=current_choice or DEFAULT_MODEL_CHOICE),
                        *update_lmstudio_options_visibility(current_choice, auto_unload),
                    )

                choices = [*MODEL_CHOICES, *lmstudio_choices]
                value = current_choice if current_choice in choices else DEFAULT_MODEL_CHOICE
                auto_col_update, auto_update = update_lmstudio_options_visibility(value, auto_unload)
                return gr.update(choices=choices, value=value), auto_col_update, auto_update

            # ===== Presets row =====
            presets_state = gr.State(get_presets())
            preset_names = sorted(list(get_presets().keys()))
            editor_visible = gr.State(False)
            initial_preset_name = preset_names[0] if preset_names else None
            initial_preset_text = get_presets().get(initial_preset_name, "") if initial_preset_name else ""

            with gr.Row(
                elem_id=self.elem_id("mp_preset_bar"),
                elem_classes=["mp-compact-row", "mp-primary-control-row", "mp-preset-bar"],
            ):
                with gr.Column(scale=1, min_width=260):
                    header_presets = gr.Dropdown(
                        choices=preset_names,
                        value=initial_preset_name,
                        label="", show_label=False,
                    )
                with gr.Column(scale=1, min_width=260):
                    with gr.Row(elem_classes=["mp-two-column-grid", "mp-preset-actions"]):
                        edit_btn = gr.Button("Edit Presets", elem_classes=["mp-rounded-btn"])
                        refresh_presets_btn = gr.Button("Refresh", elem_classes=["mp-rounded-btn"])

            # Modal editor
            with gr.Box(
                visible=False,
                elem_id=self.elem_id("mp_preset_modal"),
                elem_classes=["mp-preset-modal"],
            ) as preset_editor:
                with gr.Box(
                    elem_id=self.elem_id("mp_preset_modal_panel"),
                    elem_classes=["mp-preset-modal-panel"],
                ):
                    gr.Markdown("### Preset Editor")
                    editor_select = gr.Dropdown(
                        choices=preset_names,
                        value=initial_preset_name,
                        label="Preset",
                    )
                    editor_name = gr.Textbox(label="Name", value=initial_preset_name or "")
                    editor_text = gr.Textbox(
                        label="Text",
                        value=initial_preset_text,
                        lines=8,
                        elem_id=self.elem_id("mp_preset_editor_text"),
                        elem_classes=["mp-preset-editor-text"],
                    )
                    with gr.Row(
                        elem_id=self.elem_id("mp_preset_modal_actions"),
                        elem_classes=["mp-preset-modal-actions"],
                    ):
                        new_btn = gr.Button("New", elem_classes=["mp-rounded-btn"])
                        save_btn = gr.Button("Save", elem_classes=["mp-rounded-btn"])
                        duplicate_btn = gr.Button("Duplicate", elem_classes=["mp-rounded-btn"])
                        delete_btn = gr.Button("Delete", elem_classes=["mp-rounded-btn"])
                        close_editor = gr.Button("Close", elem_classes=["mp-rounded-btn"])
                    status_md = gr.Markdown(visible=False)

            with gr.Row(
                elem_id=self.elem_id("mp_system_prompt_header"),
                elem_classes=["mp-system-prompt-header"],
                equal_height=False,
            ):
                gr.HTML(
                    "<span>System prompt</span>",
                    elem_classes=["mp-system-prompt-title"],
                )
                improve_prompt_enabled = gr.Checkbox(
                    label="Prompt enhancement mode (use the appropriate system prompt)",
                    value=False,
                    elem_id=self.elem_id("mp_improve_prompt_enabled"),
                    elem_classes=["mp-improve-prompt-enabled"],
                )

            with gr.Row(
                elem_id=self.elem_id("mp_system_prompt_row"),
                elem_classes=["mp-system-prompt-row"],
            ):
                prompt_text = gr.Textbox(
                    label="System prompt",
                    show_label=False,
                    value=initial_preset_text or "Describe the image",
                    lines=5,
                    scale=1,
                    elem_id=self.elem_id("mp_system_prompt"),
                    elem_classes=["mp-system-prompt"],
                )
                source_prompt_text = gr.Textbox(
                    label="Prompt to improve",
                    placeholder="Paste the prompt you want the selected model to improve",
                    show_label=False,
                    lines=5,
                    scale=1,
                    visible=False,
                    elem_id=self.elem_id("mp_source_prompt"),
                    elem_classes=["mp-system-prompt", "mp-source-prompt"],
                )

            def toggle_improve_prompt(enabled):
                if enabled:
                    return gr.update(visible=True)
                return gr.update(visible=False, value="")

            improve_prompt_enabled.change(
                fn=toggle_improve_prompt,
                inputs=[improve_prompt_enabled],
                outputs=[source_prompt_text],
            )

            model_choice.change(
                fn=update_lmstudio_options_visibility,
                inputs=[model_choice, auto_unload_lmstudio],
                outputs=[auto_unload_col, auto_unload_lmstudio],
            )

            refresh_lmstudio_models_btn.click(
                fn=refresh_lmstudio_models,
                inputs=[model_choice, auto_unload_lmstudio],
                outputs=[model_choice, auto_unload_col, auto_unload_lmstudio],
            )

            def on_select_apply(name, presets):
                presets = get_presets()
                if not name:
                    return gr.update(), "", ""
                text = presets.get(name, "")
                return gr.update(value=text), name, text

            hidden_preset_name = gr.Textbox(value=initial_preset_name or "", visible=False)
            hidden_preset_text = gr.Textbox(value=initial_preset_text, visible=False)

            header_presets.change(
                fn=on_select_apply,
                inputs=[header_presets, presets_state],
                outputs=[prompt_text, hidden_preset_name, hidden_preset_text],
            )

            def refresh_presets(curr_name, curr_text, editor_value):
                presets = get_presets()
                names = sorted(list(presets.keys()))
                value = curr_name if curr_name in presets else (names[0] if names else None)
                text = presets.get(value or "", "")
                editor_value = editor_value if editor_value in presets else value
                return (
                    presets,
                    gr.update(choices=names, value=value),
                    gr.update(choices=names, value=editor_value),
                    gr.update(value=text),
                    value or "",
                    text,
                    gr.update(visible=False, value=""),
                )

            refresh_presets_btn.click(
                fn=refresh_presets,
                inputs=[hidden_preset_name, hidden_preset_text, editor_select],
                outputs=[presets_state, header_presets, editor_select, prompt_text, hidden_preset_name, hidden_preset_text, status_md],
                show_progress=False,
            )

            def toggle_editor(vis, curr_name, curr_text, presets):
                presets = get_presets()
                names = sorted(list(presets.keys()))
                opening = not bool(vis)
                if opening:
                    if (not curr_name or curr_name not in presets) and names:
                        curr_name = names[0]
                        curr_text = presets.get(curr_name, "")
                    elif curr_name in presets:
                        curr_text = presets.get(curr_name, "")
                    return (
                        presets,
                        gr.update(choices=names, value=curr_name),
                        gr.update(visible=True),
                        True,
                        gr.update(choices=names, value=curr_name),
                        gr.update(value=curr_name),
                        gr.update(value=curr_text or presets.get(curr_name, "")),
                        gr.update(visible=False, value=""),
                    )
                else:
                    return (presets, gr.update(choices=names), gr.update(visible=False), False, gr.update(), gr.update(), gr.update(), gr.update())

            edit_btn.click(
                fn=toggle_editor,
                inputs=[editor_visible, hidden_preset_name, hidden_preset_text, presets_state],
                outputs=[presets_state, header_presets, preset_editor, editor_visible, editor_select, editor_name, editor_text, status_md],
            )
            close_editor.click(
                fn=lambda: (gr.update(visible=False), False),
                inputs=[],
                outputs=[preset_editor, editor_visible],
            )

            def unique_preset_name(presets, base_name):
                existing = set((presets or {}).keys())
                base_name = (base_name or "New preset").strip() or "New preset"
                if base_name not in existing:
                    return base_name
                idx = 2
                while f"{base_name} {idx}" in existing:
                    idx += 1
                return f"{base_name} {idx}"

            def new_preset_form(presets):
                presets = get_presets()
                name = unique_preset_name(presets, "New preset")
                return (
                    gr.update(value=None),
                    gr.update(value=name),
                    gr.update(value=""),
                    gr.update(visible=True, value=f"Enter text and click Save to create '{name}'."),
                )

            new_btn.click(
                fn=new_preset_form,
                inputs=[presets_state],
                outputs=[editor_select, editor_name, editor_text, status_md],
            )

            def editor_on_select(name, presets):
                presets = get_presets()
                txt = (presets or {}).get(name or "", "")
                return name or "", txt, gr.update(visible=False, value="")

            editor_select.change(
                fn=editor_on_select,
                inputs=[editor_select, presets_state],
                outputs=[editor_name, editor_text, status_md],
            )

            def save_preset(presets, selected_name, name, text):
                presets = get_presets()
                name = (name or "").strip()
                if not name:
                    names = sorted(list(presets.keys()))
                    return (
                        presets,
                        gr.update(choices=names),
                        gr.update(choices=names),
                        gr.update(),
                        gr.update(visible=True, value="Preset name is empty."),
                        gr.update(),
                        gr.update(),
                    )
                new = dict(presets)
                selected_name = (selected_name or "").strip()
                if selected_name and selected_name != name and selected_name in new:
                    del new[selected_name]
                new[name] = text or ""
                set_presets(new)
                names = sorted(list(new.keys()))
                return (
                    new,
                    gr.update(choices=names, value=name),
                    gr.update(choices=names, value=name),
                    gr.update(value=text or ""),
                    gr.update(visible=True, value=f"Preset '{name}' saved."),
                    name,
                    text or "",
                )

            save_btn.click(
                fn=save_preset,
                inputs=[presets_state, editor_select, editor_name, editor_text],
                outputs=[presets_state, editor_select, header_presets, prompt_text, status_md, hidden_preset_name, hidden_preset_text],
            )

            def duplicate_preset(presets, name, text):
                presets = get_presets()
                source_name = (name or "").strip() or "Preset"
                new_name = unique_preset_name(presets, f"{source_name} Copy")
                new_text = text or presets.get(source_name, "")
                presets[new_name] = new_text
                set_presets(presets)
                names = sorted(list(presets.keys()))
                return (
                    presets,
                    gr.update(choices=names, value=new_name),
                    gr.update(choices=names, value=new_name),
                    gr.update(value=new_name),
                    gr.update(value=new_text),
                    gr.update(value=new_text),
                    gr.update(visible=True, value=f"Preset '{new_name}' created."),
                    new_name,
                    new_text,
                )

            duplicate_btn.click(
                fn=duplicate_preset,
                inputs=[presets_state, editor_name, editor_text],
                outputs=[presets_state, editor_select, header_presets, editor_name, editor_text, prompt_text, status_md, hidden_preset_name, hidden_preset_text],
            )

            def delete_preset(presets, name, _):
                presets = get_presets()
                name = (name or "").strip()
                new = dict(presets)
                msg = ""
                if name in new:
                    del new[name]
                    set_presets(new)
                    msg = f"Preset '{name}' deleted."
                else:
                    msg = f"Preset '{name}' not found."
                names = sorted(list(new.keys()))
                new_value = names[0] if names else None
                new_text = new.get(new_value or "", "")
                return (
                    new,
                    gr.update(choices=names, value=new_value),
                    gr.update(choices=names, value=new_value),
                    gr.update(value=new_value or ""),
                    gr.update(value=new_text),
                    gr.update(value=new_text),
                    gr.update(visible=True, value=msg),
                    new_value or "",
                    new_text,
                )

            delete_btn.click(
                fn=delete_preset,
                inputs=[presets_state, editor_name, editor_select],
                outputs=[presets_state, editor_select, header_presets, editor_name, editor_text, prompt_text, status_md, hidden_preset_name, hidden_preset_text],
            )

            # ===== Images: toolbar + paste button + drop zone + gallery =====
            images_state = gr.State([])

            with gr.Row(
                elem_id=self.elem_id("mp_upload_bar"),
                elem_classes=["mp-action-grid", "mp-upload-bar"],
            ):
                paste_btn = gr.Button(
                    "Paste from clipboard",
                    elem_id=self.elem_id("mp_paste_btn"),
                    elem_classes=["mp-rounded-btn"],
                )
                remove_last_btn = gr.Button("Remove last", elem_classes=["mp-rounded-btn"])
                clear_btn = gr.Button("Clear all", elem_classes=["mp-rounded-btn"])

            paste_pipe = gr.Textbox(
                visible=False,
                elem_id=self.elem_id("mp_paste_pipe"),
            )

            paste_btn.click(
                fn=None,
                inputs=[],
                outputs=[paste_pipe],
                _js="""
                async () => {
                  if (!(navigator.clipboard && navigator.clipboard.read)) { return ""; }
                  try{
                    const items = await navigator.clipboard.read();
                    const urls = [];
                    for (const item of items){
                      for (const type of item.types){
                        if (type.startsWith('image/')){
                          const blob = await item.getType(type);
                          const dataUrl = await new Promise(res=>{
                            const r=new FileReader(); r.onload=()=>res(r.result); r.readAsDataURL(blob);
                          });
                          urls.push(dataUrl);
                        }
                      }
                    }
                    return JSON.stringify(urls);
                  }catch(e){ console.warn(e); return ""; }
                }
                """,
            )

            drop_zone = gr.File(
                label="",
                show_label=False,
                file_types=["image"],
                file_count="multiple",
                elem_id=self.elem_id("mp_drop"),
                elem_classes=["mp-drop"],
            )

            delete_pipe_elem_id = self.elem_id("mp_delete_pipe")

            # Custom gallery with delete buttons
            with gr.Box(
                elem_id=self.elem_id("mp_gallery_container"),
                elem_classes=["mp-gallery-container"],
            ):
                gallery_html = gr.HTML(
                    value="",
                    elem_id=self.elem_id("mp_custom_gallery"),
                    elem_classes=["mp-custom-gallery"],
                )

            # Hidden textbox for receiving delete index from JS
            delete_index_pipe = gr.Textbox(
                visible=True,
                show_label=False,
                elem_id=delete_pipe_elem_id,
                elem_classes=["mp-delete-pipe-class"],
            )

            # Also keep invisible standard gallery for compatibility
            gallery_compat = gr.Gallery(visible=False)

            def render_gallery(images):
                if not images:
                    return ""

                html_parts = ['<div class="mp-thumbnails">']

                for idx, img in enumerate(images):
                    # Convert PIL image to base64 for display
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    data_url = f"data:image/jpeg;base64,{b64}"

                    html_parts.append(f'''
                    <div class="thumbnail-item">
                        <img src="{data_url}" />
                        <button type="button"
                                class="mp-delete-btn"
                                data-mp-delete-index="{idx}"
                                data-mp-delete-pipe-id="{delete_pipe_elem_id}"
                                aria-label="Remove image">&times;</button>
                    </div>
                    ''')

                html_parts.append('</div>')

                return ''.join(html_parts)

            def add_to_state(existing, new_files):
                items = list(existing or [])
                if new_files:
                    for f in new_files:
                        try:
                            im = Image.open(f.name).convert("RGB")
                            items.append(im)
                        except Exception:
                            pass
                if len(items) > MAX_IMAGES:
                    items = items[:MAX_IMAGES]
                return items, render_gallery(items), gr.update(value=None)

            drop_zone.change(
                fn=add_to_state,
                inputs=[images_state, drop_zone],
                outputs=[images_state, gallery_html, drop_zone],
            )

            def ingest_paste(existing, payload_json):
                items = list(existing or [])
                try:
                    arr = json.loads(payload_json or "[]")
                except Exception:
                    arr = []
                for data_url in arr:
                    try:
                        comma = data_url.find(",")
                        b64 = data_url[comma + 1 :] if comma != -1 else data_url
                        raw = base64.b64decode(b64)
                        im = Image.open(io.BytesIO(raw)).convert("RGB")
                        items.append(im)
                    except Exception:
                        pass
                if len(items) > MAX_IMAGES:
                    items = items[:MAX_IMAGES]
                return items, render_gallery(items), gr.update(value="")

            paste_pipe.change(
                fn=ingest_paste,
                inputs=[images_state, paste_pipe],
                outputs=[images_state, gallery_html, paste_pipe],
            )

            def delete_image_at_index(existing, index_str):
                items = list(existing or [])
                try:
                    idx = int(index_str)
                    if 0 <= idx < len(items):
                        items.pop(idx)
                except Exception:
                    pass
                return items, render_gallery(items), ""

            delete_index_pipe.change(
                fn=delete_image_at_index,
                inputs=[images_state, delete_index_pipe],
                outputs=[images_state, gallery_html, delete_index_pipe],
            )

            def remove_last(existing):
                items = list(existing or [])
                if items:
                    items.pop()
                return items, render_gallery(items)

            remove_last_btn.click(
                fn=remove_last,
                inputs=[images_state],
                outputs=[images_state, gallery_html],
            )

            def clear_all():
                return [], "", gr.update(value="")

            clear_btn.click(
                fn=clear_all,
                inputs=[],
                outputs=[images_state, gallery_html, paste_pipe],
            )

            # ===== Extra tune =====
            with gr.Row():
                append_text = gr.Textbox(label="Append to generated prompt", placeholder="e.g. in Van Gogh style, 8K resolution")
            with gr.Row():
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")
                max_tokens = gr.Slider(1, 32768, value=4096, step=1, label="Max tokens")
                top_p = gr.Slider(0.0, 1.0, value=1.0, step=0.1, label="Top P")
            sampling_notice = gr.Markdown(
                "Temperature and Top P are not supported by this Gemini model and will not be sent.",
                visible=False,
            )

            def update_sampling_controls(model_name):
                provider, model = normalize_model_choice(model_name)
                supported = not (
                    provider == "gemini"
                    and model in GEMINI_MODELS_WITHOUT_SAMPLING
                )
                return (
                    gr.update(interactive=supported),
                    gr.update(interactive=supported),
                    gr.update(visible=not supported),
                )

            model_choice.change(
                fn=update_sampling_controls,
                inputs=[model_choice],
                outputs=[temperature, top_p, sampling_notice],
                show_progress=False,
            )

            # ===== Model I/O =====
            mistral_output = gr.Textbox(
                label="Prompt from model",
                lines=4,
                elem_id=self.elem_id("mp_model_output"),
                elem_classes=["mp-model-output"],
            )
            with gr.Row(
                elem_id=self.elem_id("mp_output_actions"),
                elem_classes=["mp-two-column-grid", "mp-output-actions"],
            ):
                get_prompt_btn = gr.Button("Get Prompt", elem_classes=["mp-rounded-btn"])
                insert_btn = gr.Button("Insert into Prompt", elem_classes=["mp-rounded-btn"])

            def fetch_prompt(model_name, auto_unload, images, init_prompt, source_prompt, append, temp, max_toks, t_p):
                try:
                    provider, normalized_model = normalize_model_choice(model_name)
                    request_prompt = build_prompt_for_request(init_prompt, source_prompt)
                    text = send_to_selected_model(model_name, request_prompt, images, temp, max_toks, t_p)
                    if (append or "").strip():
                        text = f"{text}, {append.strip()}"
                    if auto_unload and provider == "lmstudio":
                        try:
                            unload_lmstudio_model(normalized_model)
                        except Exception as unload_error:
                            message = f"LM Studio auto-unload failed: {unload_error}"
                            print(f"Mistral++: {message}")
                            if hasattr(gr, "Warning"):
                                gr.Warning(message)
                    return text
                except Exception as e:
                    return f"Error: {e}"

            get_prompt_btn.click(
                fn=fetch_prompt,
                inputs=[model_choice, auto_unload_lmstudio, images_state, prompt_text, source_prompt_text, append_text, temperature, max_tokens, top_p],
                outputs=[mistral_output],
            )

            if is_img2img:
                insert_js = """
                (p) => { const app = gradioApp?.(); const ta = app?.querySelector('#img2img_prompt textarea') || app?.querySelector('[data-testid="img2img_prompt"] textarea'); if (ta){ ta.value=p||''; ta.dispatchEvent(new Event('input',{bubbles:true})); ta.dispatchEvent(new Event('change',{bubbles:true})); ta.focus(); } return p; }
                """
            else:
                insert_js = """
                (p) => { const app = gradioApp?.(); const ta = app?.querySelector('#txt2img_prompt textarea') || app?.querySelector('[data-testid="txt2img_prompt"] textarea'); if (ta){ ta.value=p||''; ta.dispatchEvent(new Event('input',{bubbles:true})); ta.dispatchEvent(new Event('change',{bubbles:true})); ta.focus(); } return p; }
                """
            insert_btn.click(fn=None, inputs=[mistral_output], outputs=[mistral_output], _js=insert_js)

        return [mistral_output]

    def run(self, p, *args):
        return processing.process_images(p)

# ========= Settings =========

def on_ui_settings():
    section = ("mistral_prompt", "Mistral++")
    extension_setting_keys = (
        "mistral_api_key",
        "GEMINI_API_KEY",
        "groq_api_key",
        "lmstudio_api_base",
        "lmstudio_api_key",
        "mistral_image_max_size",
        "mistral_image_max_kb",
    )

    shared.opts.add_option(
        "mistral_api_key",
        shared.OptionInfo("", "Mistral API Key", section=section).html(
            "[<a href='https://admin.mistral.ai/organization/api-keys' "
            "target='_blank'>Get API key</a>]"
        )
    )

    shared.opts.add_option(
        "GEMINI_API_KEY",
        shared.OptionInfo("", "Gemini API Key", section=section).html(
            "[<a href='https://aistudio.google.com/api-keys' "
            "target='_blank'>Get API key</a>]"
        )
    )

    shared.opts.add_option(
        "groq_api_key",
        shared.OptionInfo("", "Groq API Key", section=section).html(
            "[<a href='https://console.groq.com/keys' "
            "target='_blank'>Get API key</a>]"
        )
    )

    shared.opts.add_option(
        "lmstudio_api_base",
        shared.OptionInfo(
            LMSTUDIO_DEFAULT_API_BASE,
            "LM Studio API Base",
            section=section
        )
    )

    shared.opts.add_option(
        "lmstudio_api_key",
        shared.OptionInfo("", "LM Studio API Key (optional)", section=section)
    )

    shared.opts.add_option(
        "mistral_image_max_size",
        shared.OptionInfo(
            768,
            "Max image size sent to model (longest side, px)",
            section=section
        )
    )

    shared.opts.add_option(
        "mistral_image_max_kb",
        shared.OptionInfo(
            400,
            "Max JPEG size sent to model (KB)",
            section=section
        )
    )

    # Reassigning an existing dict key does not change its position. This matters
    # after extension reloads, where a newly introduced option would otherwise be
    # appended below all previously registered settings.
    registered = {}
    for key in extension_setting_keys:
        info = shared.opts.data_labels.pop(key, None)
        if info is not None:
            registered[key] = info
    shared.opts.data_labels.update(registered)

try:
    from modules import script_callbacks
    script_callbacks.on_ui_settings(on_ui_settings)
except Exception:
    pass
