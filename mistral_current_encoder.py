"""Generate text with Forge's selected Qwen 3VL 4B or 8B text encoder.

This module reads Forge model objects but does not modify them or Forge files.
"""

import numpy as np
import torch
from PIL import ImageOps


STOP_TOKENS = {151643, 151645}


def build_prompt_tokens(tokenizer, prompt, images):
    """Build the Qwen chat prompt, leaving image payloads for Forge's CLIP wrapper."""
    def ids(text):
        return list(tokenizer(text)["input_ids"])

    tokens = ids("<|im_start|>user\n")
    for image in images:
        tokens.extend(ids("<|vision_start|>"))
        tokens.append({"type": "image", "data": image, "original_type": "image"})
        tokens.extend(ids("<|vision_end|>"))
    tokens.extend(ids(prompt.strip() + "<|im_end|>\n<|im_start|>assistant\n"))
    return tokens


def get_qwen3vl(sd_model):
    clip = getattr(getattr(sd_model, "forge_objects", None), "clip", None)
    text_models = getattr(clip, "cond_stage_model", None)
    tokenizers = getattr(clip, "tokenizer", None)
    engine = getattr(sd_model, "text_processing_engine_qwen", None)
    encoder = getattr(engine, "text_encoder", None)
    transformer = getattr(encoder, "transformer", None)
    model_type = next(
        (name for name in ("qwen3vl_4b", "qwen3vl_8b")
         if transformer is not None and getattr(text_models, name, None) is transformer
         and getattr(tokenizers, name, None) is not None),
        None,
    )
    if model_type is None or not hasattr(encoder, "process_tokens"):
        raise ValueError("Current model text encoder supports Qwen 3VL 4B and 8B only.")
    tokenizer = getattr(tokenizers, model_type)
    core = getattr(transformer, "model", None)
    if core is None or not hasattr(core, "layers") or not hasattr(core, "embed_tokens"):
        raise ValueError("Current Qwen 3VL encoder has no text generation weights.")
    if model_type == "qwen3vl_8b" and not hasattr(core, "lm_head"):
        raise ValueError("Current Qwen 3VL 8B encoder has no output head.")
    return clip, encoder, tokenizer, transformer, core


def get_selected_sd_model(model_data, reload_model):
    """Use the model selected in Forge, loading it if the selection has changed."""
    if model_data.forge_hash != str(model_data.forge_loading_parameters):
        sd_model, _ = reload_model()
        return sd_model
    return model_data.sd_model


def sample_token(logits, temperature, top_p):
    """Sample one token; zero temperature selects the highest logit."""
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1:
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        discard = torch.cumsum(sorted_probs, dim=-1) - sorted_probs >= top_p
        sorted_probs[discard] = 0
        sorted_probs /= sorted_probs.sum()
        return int(sorted_indices[torch.multinomial(sorted_probs, 1)].item())
    return int(torch.multinomial(probabilities, 1).item())


def _image_tensor(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((768, 768))
    pixels = np.asarray(image).copy()
    return torch.from_numpy(pixels).unsqueeze(0).float().div_(255)


def _forward_step(core, embeddings, positions, cache, visual_mask=None, deepstack=None):
    from backend.nn.llm.llama import attention_function, precompute_freqs_cis

    config = core.config
    x = embeddings
    if core.normalize_in:
        x = x * (config.hidden_size ** 0.5)
    freqs = precompute_freqs_cis(
        config.head_dim, positions, config.rope_theta, config.rope_scale,
        config.rope_dims, interleaved_mrope=getattr(config, "interleaved_mrope", False),
        device=x.device,
    )
    mask = None
    if x.shape[1] > 1:
        mask = torch.empty(x.shape[1], x.shape[1], device=x.device, dtype=x.dtype)
        mask.fill_(torch.finfo(x.dtype).min / 4).triu_(1)

    next_cache = []
    for index, layer in enumerate(core.layers):
        x, state = layer(
            x, attention_mask=mask, freqs_cis=freqs,
            optimized_attention=attention_function,
            past_key_value=cache[index],
        )
        if deepstack is not None and index < len(deepstack):
            x[visual_mask] = x[visual_mask] + deepstack[index].to(x)
        next_cache.append(state)
    if core.norm is not None:
        x = core.norm(x)
    return x, next_cache


def _reserve_cache(cache, additional_tokens):
    reserved = []
    for key, value, used in cache:
        shape = (*key.shape[:2], used + additional_tokens, key.shape[-1])
        keys = torch.empty(shape, device=key.device, dtype=key.dtype)
        values = torch.empty(shape, device=value.device, dtype=value.dtype)
        keys[:, :, :used] = key
        values[:, :, :used] = value
        reserved.append((keys, values, used))
    return reserved


def _output_weight(core, hidden):
    from backend.operations import main_stream_worker, weights_manual_cast

    embedding = core.embed_tokens
    weight, _, signal = weights_manual_cast(
        embedding, dtype=hidden.dtype, device=hidden.device, skip_weight_dtype=True,
    )
    with main_stream_worker(weight, None, signal):
        return weight.to(hidden.dtype)


def _token_logits(core, hidden, output_weight):
    if hasattr(core, "lm_head"):
        return core.lm_head(hidden[:, -1])[0]
    return torch.nn.functional.linear(hidden[:, -1], output_weight)[0]


def _next_token_embedding(core, token, device, output_weight):
    token_id = torch.tensor([[token]], device=device, dtype=torch.long)
    if hasattr(core, "lm_head"):
        return core.embed_tokens(token_id)
    return torch.nn.functional.embedding(token_id, output_weight)


@torch.inference_mode()
def generate_current_prompt(prompt, images, temperature, maximum_tokens, top_p):
    from backend import memory_management
    from modules import sd_models

    sd_model = get_selected_sd_model(sd_models.model_data, sd_models.forge_model_reload)
    clip, encoder, tokenizer, transformer, core = get_qwen3vl(sd_model)
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is empty.")
    if not 0 < int(maximum_tokens) <= 32768:
        raise ValueError("Max tokens must be between 1 and 32768.")
    maximum_tokens = int(maximum_tokens)

    memory_management.load_model_gpu(clip.patcher)
    device = clip.patcher.load_device
    image_tensors = [_image_tensor(image) for image in (images or [])]
    tokens = build_prompt_tokens(tokenizer, prompt, image_tensors)
    embeddings, _, _, embeds_info = encoder.process_tokens([tokens], device)
    positions, visual_mask, deepstack = transformer.build_image_inputs(embeddings, embeds_info)
    if positions is None:
        positions = torch.arange(embeddings.shape[1], device=device).unsqueeze(0)
    next_position = int(positions[:, -1].max().item()) + 1
    if device.type == "cuda":
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        compute_dtype = torch.float32
    embeddings = embeddings.to(compute_dtype)

    hidden, cache = _forward_step(
        core, embeddings, positions, [()] * len(core.layers), visual_mask, deepstack,
    )
    cache = _reserve_cache(cache, min(maximum_tokens, 256))
    output_weight = None if hasattr(core, "lm_head") else _output_weight(core, hidden)
    generated = []
    progress_label = "Mistral++ current encoder"
    print(f"{progress_label}: 0/{maximum_tokens} tokens", end="", flush=True)
    try:
        for step in range(maximum_tokens):
            logits = _token_logits(core, hidden, output_weight)
            token = sample_token(logits, float(temperature), float(top_p))
            if token in STOP_TOKENS:
                break
            generated.append(token)
            print(f"\r{progress_label}: {len(generated)}/{maximum_tokens} tokens", end="", flush=True)
            if step + 1 == maximum_tokens:
                break
            if cache[0][2] >= cache[0][0].shape[2]:
                cache = _reserve_cache(cache, min(maximum_tokens - step - 1, 256))
            next_embed = _next_token_embedding(core, token, device, output_weight).to(compute_dtype)
            next_positions = torch.tensor([[next_position]], device=device, dtype=torch.long)
            hidden, cache = _forward_step(core, next_embed, next_positions, cache)
            next_position += 1
    finally:
        print()

    result = tokenizer.decode(generated, skip_special_tokens=True).rpartition("</think>")[-1].strip()
    if not result:
        raise ValueError("Current model text encoder returned an empty response.")
    return result
