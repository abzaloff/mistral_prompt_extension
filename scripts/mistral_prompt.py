# extensions/<your_extension>/scripts/mistral_prompt.py
# Minimal UI + clipboard button only:
# - Drag&drop / click / "Paste from clipboard" button
# - Remove last / Clear all / Individual delete buttons
# - Presets + inline editor
# - Send to Mistral API + insert into main prompt

import io
import json
import base64
import requests
from PIL import Image

import gradio as gr
from modules import scripts, shared, processing

MAX_IMAGES = 30
API_URL = "https://api.mistral.ai/v1/chat/completions"

# ========= Presets =========
PRESETS_OPT_KEY = "mistral_presets_json"
DEFAULT_PRESETS = {
    "Flux - Describe": "Describe the image",
    "SDXL - Tokens": "Describe the image using only comma-separated tokens",
}

def _ensure_presets_in_opts():
    raw = shared.opts.data.get(PRESETS_OPT_KEY, "").strip()
    if not raw:
        shared.opts.data[PRESETS_OPT_KEY] = json.dumps(DEFAULT_PRESETS, ensure_ascii=False)
        try:
            shared.opts.save(shared.config_filename)
        except Exception:
            pass

def get_presets():
    _ensure_presets_in_opts()
    raw = shared.opts.data.get(PRESETS_OPT_KEY, "{}")
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else dict(DEFAULT_PRESETS)
    except Exception:
        return dict(DEFAULT_PRESETS)

def set_presets(presets: dict):
    shared.opts.data[PRESETS_OPT_KEY] = json.dumps(presets, ensure_ascii=False)
    try:
        shared.opts.save(shared.config_filename)
    except Exception:
        pass

# ========= Mistral API =========

# Reuse one HTTP session to persist cookies (helps with Cloudflare checks).
_mistral_session = None

def get_mistral_session():
    global _mistral_session
    if _mistral_session is None:
        _mistral_session = requests.Session()
    return _mistral_session

def send_to_mistral(prompt, images, temperature, maximum_tokens, top_p):
    api_key = shared.opts.data.get("mistral_api_key", "").strip()
    if not api_key:
        raise ValueError("Mistral API key is not set in Settings.")

    image_urls = []
    for img in images:
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

        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_urls.append(f"data:image/jpeg;base64,{b64}")

    if len(image_urls) > MAX_IMAGES:
        raise ValueError(f"Maximum {MAX_IMAGES} images supported.")

    content_list = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content_list.append({"type": "image_url", "image_url": url})

    data = {
        "model": "pixtral-large-latest",
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

# ========= UI Script =========

class Script(scripts.Script):
    def title(self):
        return "Mistral Prompt"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        _ensure_presets_in_opts()

        with gr.Accordion("Mistral Prompt", open=False):

            # ===== CSS + JS: dropzone visuals + delete buttons =====
            gr.HTML(
                """
<style>
  /* presets */
  #mp_preset_bar{display:flex;gap:8px;align-items:flex-end;flex-wrap:nowrap}
  #mp_preset_bar label{display:none !important;}
  #mp_preset_bar .gr-form{margin-bottom:0 !important;}
  #mp_preset_bar .gr-dropdown, #mp_preset_bar .wrap{min-width:240px;}
  #mp_preset_bar .gr-button{white-space:nowrap}

  /* upload toolbar: three equal buttons */
  #mp_upload_bar{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;align-items:stretch}
  #mp_upload_bar .gr-button{width:100%}

  /* fixed-height drop zone to avoid layout jumps while uploading */
  #mp_drop{position:relative;isolation:isolate;margin-top:6px;margin-bottom:0;min-height:84px !important;height:84px !important;overflow:hidden;}

  #mp_drop .wrap,
  #mp_drop .file-wrap,
  #mp_drop .border,
  #mp_drop .container{
    height:100% !important;
    min-height:100% !important;
    padding:0 !important;
    background:transparent !important;
    border:none !important;
  }

  /* keep upload status/progress visible */
  #mp_drop [class*="status"],
  #mp_drop [class*="progress"],
  #mp_drop [data-testid*="status"],
  #mp_drop [data-testid*="progress"]{
      position:absolute;
      bottom:6px;
      right:10px;
      z-index:4;
      font-size:12px;
      opacity:.9 !important;
      visibility:visible !important;
  }

  /* hide default Gradio hints in gr.File without breaking click handling */
  #mp_drop label,
  #mp_drop .label,
  #mp_drop .upload-text,
  #mp_drop .filetype,
  #mp_drop p,
  #mp_drop span{
    opacity:0 !important;
  }

  /* custom drop-zone label shown as an overlay */
  #mp_drop::after{
      content:"Drag images here or click to select, or use the \\"Paste from clipboard\\" button";
      position:absolute;
      inset:0;
      display:flex;
      align-items:center;
      justify-content:center;
      padding:0 14px;
      font-size:13.5px;
      font-weight:600;
      opacity:.95;
      border:1.5px dashed var(--block-border-color);
      border-radius:8px;
      background:var(--body-background-fill);
      text-align:center;
      pointer-events:none;
      z-index:5; /* keep overlay above default Gradio text */
  }
  #mp_drop.dragover::after{ content:"Drop to add images"; }

  /* gallery with delete buttons */
  #mp_gallery_container{position:relative;margin-top:8px;}
  #mp_gallery .thumbnails{display:grid !important;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;}
  #mp_gallery .thumbnail-item{position:relative;aspect-ratio:1;border-radius:8px;overflow:hidden;}
  #mp_gallery .thumbnail-item img{width:100%;height:100%;object-fit:cover;}

  /* delete button */
  .mp-delete-btn{
    position:absolute;top:4px;right:4px;z-index:10;
    width:24px;height:24px;border-radius:50%;
    background:rgba(0,0,0,0.7);color:#fff;
    border:none;cursor:pointer;
    display:flex;align-items:center;justify-content:center;
    padding:0;
    font-size:15px;line-height:1;
    font-weight:700;
    font-family:Arial,sans-serif;
    transition:background 0.2s;
  }
  .mp-delete-btn:hover{background:rgba(220,38,38,0.9);}
  
    /* hide empty gallery container by default */
    #mp_gallery_container{
      display:none !important;
      margin:0 !important;
      padding:0 !important;
      border:none !important;
      background:transparent !important;
    }

    /* show container only when at least one image exists */
    #mp_gallery_container:has(img){
      display:block !important;
      margin-top:8px !important; /* controlled gap when preview appears */
    }
</style>

<script>
(function(){
  function appRoot(){ try{ return window.gradioApp ? gradioApp() : document; }catch(e){ return document; } }

  // Global helper to delete an image by index.
  window.deleteMPImage = function(idx) {
    const app = appRoot();
    const pipe = app.querySelector('#mp_delete_pipe textarea');
    if (pipe) {
      pipe.value = idx.toString();
      pipe.dispatchEvent(new Event('input', {bubbles: true}));
    }
  };

  function setupDragOnly(){
    const app = appRoot();
    const drop = app.querySelector('#mp_drop');
    if(!drop) return false;

    const prevent = e => { e.preventDefault(); e.stopPropagation(); };
    ['dragenter','dragover'].forEach(ev => drop.addEventListener(ev, e => { prevent(e); drop.classList.add('dragover'); }));
    ['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => { prevent(e); drop.classList.remove('dragover'); }));

    const ensureHeights = () => {
      const boxes = drop.querySelectorAll('.wrap, .file-wrap, .border, .container');
      boxes.forEach(b=>{ b.style.height='100%'; b.style.minHeight='100%'; });
    };
    ensureHeights();
    new MutationObserver(ensureHeights).observe(drop, {subtree:true, childList:true, attributes:true});

    return true;
  }

  let tries = 0;
  const t = setInterval(() => { if (setupDragOnly() || ++tries > 120) clearInterval(t); }, 100);
})();
</script>
"""
            )

            # ===== Presets row =====
            presets_state = gr.State(get_presets())
            preset_names = sorted(list(get_presets().keys()))
            editor_visible = gr.State(False)

            with gr.Row(elem_id="mp_preset_bar"):
                header_presets = gr.Dropdown(
                    choices=preset_names,
                    value=preset_names[0] if preset_names else None,
                    label="", show_label=False,
                )
                edit_btn = gr.Button("Edit")

            # Inline editor
            with gr.Box(visible=False) as preset_editor:
                gr.Markdown("### Preset Editor")
                with gr.Row():
                    editor_select = gr.Dropdown(choices=preset_names, value=preset_names[0] if preset_names else None, label="Select preset")
                    close_editor = gr.Button("Close")
                editor_name = gr.Textbox(label="Preset name")
                editor_text = gr.Textbox(label="Preset text", lines=4)
                with gr.Row():
                    save_btn = gr.Button("Save / Update")
                    delete_btn = gr.Button("Delete")
                status_md = gr.Markdown(visible=False)

            with gr.Row():
                prompt_text = gr.Textbox(label="Initial prompt for Mistral", value="Describe the image")

            def on_select_apply(name, presets):
                if not name:
                    return gr.update(), "", ""
                text = presets.get(name, "")
                return gr.update(value=text), name, text

            hidden_preset_name = gr.Textbox(visible=False)
            hidden_preset_text = gr.Textbox(visible=False)

            header_presets.change(
                fn=on_select_apply,
                inputs=[header_presets, presets_state],
                outputs=[prompt_text, hidden_preset_name, hidden_preset_text],
            )

            def toggle_editor(vis, curr_name, curr_text, presets):
                presets = dict(presets)
                names = sorted(list(presets.keys()))
                opening = not bool(vis)
                if opening:
                    if not curr_name and names:
                        curr_name = names[0]
                        curr_text = presets.get(curr_name, "")
                    return (
                        gr.update(visible=True),
                        True,
                        gr.update(choices=names, value=curr_name),
                        gr.update(value=curr_name),
                        gr.update(value=curr_text or presets.get(curr_name, "")),
                    )
                else:
                    return (gr.update(visible=False), False, gr.update(), gr.update(), gr.update())

            edit_btn.click(
                fn=toggle_editor,
                inputs=[editor_visible, hidden_preset_name, hidden_preset_text, presets_state],
                outputs=[preset_editor, editor_visible, editor_select, editor_name, editor_text],
            )
            close_editor.click(
                fn=lambda: (gr.update(visible=False), False),
                inputs=[],
                outputs=[preset_editor, editor_visible],
            )

            def editor_on_select(name, presets):
                txt = (presets or {}).get(name or "", "")
                return name or "", txt

            editor_select.change(
                fn=editor_on_select,
                inputs=[editor_select, presets_state],
                outputs=[editor_name, editor_text],
            )

            def save_preset(presets, name, text):
                name = (name or "").strip()
                if not name:
                    names = sorted(list(presets.keys()))
                    return presets, gr.update(choices=names), gr.update(choices=names), "Preset name is empty.", hidden_preset_name, hidden_preset_text
                new = dict(presets)
                new[name] = text or ""
                set_presets(new)
                names = sorted(list(new.keys()))
                return new, gr.update(choices=names, value=name), gr.update(choices=names, value=name), f"Preset '{name}' saved.", name, text

            save_btn.click(
                fn=save_preset,
                inputs=[presets_state, editor_name, editor_text],
                outputs=[presets_state, editor_select, header_presets, status_md, hidden_preset_name, hidden_preset_text],
            )

            def delete_preset(presets, name, _):
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
                return new, gr.update(choices=names, value=new_value), gr.update(choices=names, value=new_value), msg, new_value or "", new.get(new_value or "", "")

            delete_btn.click(
                fn=delete_preset,
                inputs=[presets_state, editor_name, editor_select],
                outputs=[presets_state, editor_select, header_presets, status_md, hidden_preset_name, hidden_preset_text],
            )

            # ===== Images: toolbar + paste button + drop zone + gallery =====
            images_state = gr.State([])

            with gr.Row(elem_id="mp_upload_bar"):
                paste_btn = gr.Button("Paste from clipboard", elem_id="mp_paste_btn")
                remove_last_btn = gr.Button("Remove last")
                clear_btn = gr.Button("Clear all")

            paste_pipe = gr.Textbox(visible=False, elem_id="mp_paste_pipe")

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

            drop_zone = gr.File(label="", show_label=False, file_types=["image"], file_count="multiple", elem_id="mp_drop")

            # Custom gallery with delete buttons
            with gr.Box(elem_id="mp_gallery_container"):
                gallery_html = gr.HTML(value="", elem_id="mp_custom_gallery")

            # Hidden textbox for receiving delete index from JS
            delete_index_pipe = gr.Textbox(visible=False, elem_id="mp_delete_pipe", elem_classes="mp-delete-pipe-class")

            # Also keep invisible standard gallery for compatibility
            gallery_compat = gr.Gallery(visible=False)

            def render_gallery(images):
                if not images:
                    return ""

                html_parts = ['<div class="mp-thumbnails" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;">']

                for idx, img in enumerate(images):
                    # Convert PIL image to base64 for display
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    data_url = f"data:image/jpeg;base64,{b64}"

                    html_parts.append(f'''
                    <div class="thumbnail-item" style="position:relative;aspect-ratio:1;border-radius:8px;overflow:hidden;">
                        <img src="{data_url}" style="width:100%;height:100%;object-fit:cover;" />
                        <button class="mp-delete-btn" 
                                onclick="(function(idx,btn){{var app=document;try{{app=gradioApp();}}catch(e){{}}var tab=btn.closest('[id*=\\'txt2img\\']')||btn.closest('[id*=\\'img2img\\']');var pipe=tab?tab.querySelector('.mp-delete-pipe-class textarea'):null;if(!pipe){{var all=app.querySelectorAll('.mp-delete-pipe-class textarea');pipe=all[0];}}if(pipe){{pipe.value=idx.toString();pipe.dispatchEvent(new Event('input',{{bubbles:true}}));}}}})({idx},this);return false;"
                                style="position:absolute;top:4px;right:4px;width:24px;height:24px;border-radius:50%;background:rgba(0,0,0,0.7);color:#fff;border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;padding:0;font-size:15px;line-height:1;font-weight:700;font-family:Arial,sans-serif;transition:background 0.2s;"
                                onmouseover="this.style.background='rgba(220,38,38,0.9)'"
                                onmouseout="this.style.background='rgba(0,0,0,0.7)'">&times;</button>
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
                append_text = gr.Textbox(label="Append to Mistral prompt", placeholder="e.g. in Van Gogh style, 8K resolution")
            with gr.Row():
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")
                max_tokens = gr.Slider(1, 32768, value=4096, step=1, label="Max tokens")
                top_p = gr.Slider(0.0, 1.0, value=1.0, step=0.1, label="Top P")

            # ===== Mistral I/O =====
            mistral_output = gr.Textbox(label="Prompt from Mistral", lines=4)
            with gr.Row():
                get_prompt_btn = gr.Button("Get Prompt from Mistral")
                insert_btn = gr.Button("Insert into Prompt")

            def fetch_prompt(images, init_prompt, append, temp, max_toks, t_p):
                if not images:
                    return "No images uploaded."
                try:
                    text = send_to_mistral(init_prompt, images, temp, max_toks, t_p)
                    if (append or "").strip():
                        text = f"{text}, {append.strip()}"
                    return text
                except Exception as e:
                    return f"Error: {e}"

            get_prompt_btn.click(
                fn=fetch_prompt,
                inputs=[images_state, prompt_text, append_text, temperature, max_tokens, top_p],
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
    section = ("mistral_prompt", "Mistral Prompt Generator")

    shared.opts.add_option(
        "mistral_api_key",
        shared.OptionInfo("", "Mistral API Key", section=section)
    )

    shared.opts.add_option(
        "mistral_image_max_size",
        shared.OptionInfo(
            768,
            "Max image size sent to Mistral (longest side, px)",
            section=section
        )
    )

    shared.opts.add_option(
        "mistral_image_max_kb",
        shared.OptionInfo(
            400,
            "Max JPEG size sent to Mistral (KB)",
            section=section
        )
    )

    shared.opts.add_option(
        PRESETS_OPT_KEY,
        shared.OptionInfo(
            json.dumps(DEFAULT_PRESETS, ensure_ascii=False),
            "Mistral Presets (JSON)",
            section=section
        ),
    )

try:
    from modules import script_callbacks
    script_callbacks.on_ui_settings(on_ui_settings)
except Exception:
    pass
