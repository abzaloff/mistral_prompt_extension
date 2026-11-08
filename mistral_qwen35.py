"""Text generation with Forge's loaded Qwen3.5 9B encoder (MOSS Image)."""

from pathlib import Path

import torch
import torch.nn.functional as F


STOP_TOKENS = {248044, 248046}


def get_encoder(sd_model):
    clip = getattr(getattr(sd_model, "forge_objects", None), "clip", None)
    models = getattr(clip, "cond_stage_model", None)
    tokenizers = getattr(clip, "tokenizer", None)
    transformer = getattr(models, "qwen35_9b", None)
    tokenizer = getattr(tokenizers, "qwen35_9b", None)
    core = getattr(transformer, "model", None)
    if core is None or tokenizer is None or not hasattr(core, "layers"):
        raise ValueError("Current model text encoder supports Qwen 3VL 4B/8B and Qwen3.5 9B only.")
    return clip, tokenizer, core


def _head_path(model_data):
    """Use only a module that Forge selected for this model, never an arbitrary file."""
    from safetensors import safe_open

    paths = model_data.forge_loading_parameters.get("additional_modules", [])
    candidates = []
    for path in paths:
        path = Path(path)
        if not path.is_file() or path.suffix.lower() != ".safetensors":
            continue
        with safe_open(str(path), framework="pt", device="cpu") as weights:
            if "lm_head.weight" in weights.keys() and "model.language_model.embed_tokens.weight" in weights.keys():
                candidates.append(path)
    if len(candidates) != 1:
        raise ValueError("Qwen3.5 9B generation needs one selected text encoder module containing lm_head.weight.")
    return candidates[0]


def _load_head(path, device, dtype, vocab_size, hidden_size):
    """Load Forge's omitted output head in slices to limit host memory use."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as weights:
        source = weights.get_slice("lm_head.weight")
        if source.get_shape() != [vocab_size, hidden_size]:
            raise ValueError("Selected Qwen3.5 9B output head has an unexpected shape.")
        if source.get_dtype() not in ("BF16", "F16", "F32"):
            raise ValueError("Selected Qwen3.5 9B output head uses an unsupported quantized format.")
        head = torch.empty((vocab_size, hidden_size), device=device, dtype=dtype)
        for start in range(0, vocab_size, 2048):
            end = min(start + 2048, vocab_size)
            head[start:end].copy_(source[start:end].to(device=device, dtype=dtype))
    return head


def _linear_step(attn, x, state):
    """Advance a single GatedDeltaNet token, including its causal convolution."""
    batch, _, _ = x.shape
    proj = attn.in_proj_qkv(x).transpose(1, 2)
    conv_history, recurrent = state
    conv_input = torch.cat((conv_history, proj), dim=-1)
    mixed = F.silu(attn.conv1d(conv_input)[:, :, attn.conv_kernel_size - 1:attn.conv_kernel_size])
    next_history = conv_input[:, :, 1:]
    query, key, value = mixed.transpose(1, 2).split(
        [attn.key_dim, attn.key_dim, attn.value_dim], dim=-1
    )
    query = query.reshape(batch, attn.num_key_heads, attn.key_head_dim)
    key = key.reshape(batch, attn.num_key_heads, attn.key_head_dim)
    value = value.reshape(batch, attn.num_value_heads, attn.value_head_dim)
    if attn.num_value_heads != attn.num_key_heads:
        repeat = attn.num_value_heads // attn.num_key_heads
        query = query.repeat_interleave(repeat, dim=1)
        key = key.repeat_interleave(repeat, dim=1)
    query = F.normalize(query.float(), dim=-1) * (attn.key_head_dim ** -0.5)
    key = F.normalize(key.float(), dim=-1)
    decay = (-attn.A_log.float().exp() * F.softplus(
        attn.in_proj_a(x).float().squeeze(1) + attn.dt_bias.float()
    )).exp()
    beta = attn.in_proj_b(x).float().squeeze(1).sigmoid()
    prior = recurrent * decay[..., None, None]
    residual = (value.float() - torch.einsum("bhk,bhkv->bhv", key, prior)) * beta[..., None]
    output = torch.einsum("bhk,bhkv->bhv", query, prior) + (
        (query * key).sum(-1, keepdim=True) * residual
    )
    recurrent = prior + key.unsqueeze(-1) * residual.unsqueeze(-2)
    output = output.to(x.dtype).reshape(-1, attn.value_head_dim)
    output = attn.norm(output) * F.silu(attn.in_proj_z(x).reshape(-1, attn.value_head_dim))
    output = attn.out_proj(output.reshape(batch, 1, -1))
    return output, (next_history, recurrent)


def _linear_prefill(attn, x):
    from backend.nn.llm.llama import torch_chunk_gated_delta_rule

    batch, length, _ = x.shape
    proj = attn.in_proj_qkv(x)
    mixed = F.silu(attn.conv1d(proj.transpose(1, 2))[:, :, :length]).transpose(1, 2)
    query, key, value = mixed.split([attn.key_dim, attn.key_dim, attn.value_dim], dim=-1)
    query = query.reshape(batch, length, attn.num_key_heads, attn.key_head_dim)
    key = key.reshape(batch, length, attn.num_key_heads, attn.key_head_dim)
    value = value.reshape(batch, length, attn.num_value_heads, attn.value_head_dim)
    if attn.num_value_heads != attn.num_key_heads:
        repeat = attn.num_value_heads // attn.num_key_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    decay = -attn.A_log.float().exp() * F.softplus(
        attn.in_proj_a(x).float() + attn.dt_bias.float()
    )
    beta = attn.in_proj_b(x).sigmoid()
    output, recurrent = torch_chunk_gated_delta_rule(
        query, key, value, decay, beta, output_final_state=True
    )
    output = output.reshape(-1, attn.value_head_dim)
    output = attn.norm(output) * F.silu(attn.in_proj_z(x).reshape(-1, attn.value_head_dim))
    output = attn.out_proj(output.reshape(batch, length, -1))
    history = F.pad(proj.transpose(1, 2), (attn.conv_kernel_size - 1, 0))[:, :, -(attn.conv_kernel_size - 1):]
    return output, (history, recurrent)


def _attention_step(attn, x, freqs, cache):
    from backend.nn.llm.llama import apply_partial_rope, attention_function

    batch, length, _ = x.shape
    qg = attn.q_proj(x).view(batch, length, attn.num_heads, attn.head_dim * 2)
    query, gate = qg[..., :attn.head_dim], qg[..., attn.head_dim:]
    query = attn.q_norm(query).transpose(1, 2)
    key = attn.k_norm(attn.k_proj(x).view(batch, length, attn.num_kv_heads, attn.head_dim)).transpose(1, 2)
    value = attn.v_proj(x).view(batch, length, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    query, key = apply_partial_rope(query, key, freqs, attn.rotary_dim)
    if cache is not None:
        key = torch.cat((cache[0], key), dim=2)
        value = torch.cat((cache[1], value), dim=2)
    mask = None
    if length > 1:
        mask = torch.empty(length, length, device=x.device, dtype=x.dtype).fill_(torch.finfo(x.dtype).min / 4).triu_(1)
    output = attention_function(query, key, value, attn.num_heads, mask=mask, skip_reshape=True,
                                enable_gqa=attn.num_heads != attn.num_kv_heads)
    output = output * gate.reshape(batch, length, -1).sigmoid()
    return attn.o_proj(output), (key, value)


def forward_cached(core, embeddings, start_position, states=None):
    """Run Qwen3.5 blocks using the already loaded Forge weights and local cache."""
    from backend.nn.llm.llama import precompute_freqs_cis

    length = embeddings.shape[1]
    positions = torch.arange(start_position, start_position + length, device=embeddings.device).unsqueeze(0)
    config = core.config
    freqs = precompute_freqs_cis(
        int(config.head_dim * config.partial_rotary_factor), positions,
        config.rope_theta, config.rope_scale, config.mrope_section,
        interleaved_mrope=True, device=embeddings.device,
    )
    x = embeddings
    new_states = []
    for index, layer in enumerate(core.layers):
        normalized = layer.input_layernorm(x)
        if layer.layer_type == "linear_attention":
            if states is None:
                out, state = _linear_prefill(layer.linear_attn, normalized)
            else:
                out, state = _linear_step(layer.linear_attn, normalized, states[index])
        else:
            out, state = _attention_step(layer.self_attn, normalized, freqs,
                                         None if states is None else states[index])
        x = x + out
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        new_states.append(state)
    if core.norm is not None:
        x = core.norm(x)
    return x, new_states


@torch.inference_mode()
def generate(prompt, images, temperature, maximum_tokens, top_p, model_data, sd_model, sample_token):
    from backend import memory_management

    if images:
        raise ValueError("Qwen3.5 9B in Forge supports text input only; remove attached images.")
    clip, tokenizer, core = get_encoder(sd_model)
    memory_management.load_model_gpu(clip.patcher)
    device = clip.patcher.load_device
    engine = sd_model.text_processing_engine_qwen
    input_ids = engine.tokenize(prompt.strip())
    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
    embeddings = core.embed_tokens(tokens)
    dtype = embeddings.dtype
    head = _load_head(_head_path(model_data), device, dtype, core.config.vocab_size, core.config.hidden_size)
    hidden, states = forward_cached(core, embeddings, 0)
    generated = []
    label = "Mistral++ Qwen3.5 9B"
    print(f"{label}: 0/{maximum_tokens} tokens", end="", flush=True)
    try:
        for step in range(maximum_tokens):
            logits = F.linear(hidden[:, -1].to(dtype), head)[0]
            token = sample_token(logits, float(temperature), float(top_p))
            if token in STOP_TOKENS:
                break
            generated.append(token)
            print(f"\r{label}: {len(generated)}/{maximum_tokens} tokens", end="", flush=True)
            if step + 1 < maximum_tokens:
                next_token = torch.tensor([[token]], dtype=torch.long, device=device)
                hidden, states = forward_cached(core, core.embed_tokens(next_token), len(input_ids) + step, states)
    finally:
        print()
    result = tokenizer.decode(generated, skip_special_tokens=True).rpartition("</think>")[-1].strip()
    if not result:
        raise ValueError("Current model text encoder returned an empty response.")
    return result
