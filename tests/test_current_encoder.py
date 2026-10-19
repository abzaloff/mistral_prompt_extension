import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "mistral_current_encoder.py"
SPEC = importlib.util.spec_from_file_location("mistral_current_encoder", MODULE_PATH)
current_encoder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(current_encoder)


class Tokenizer:
    def __call__(self, text):
        return {"input_ids": [ord(char) for char in text]}


class CurrentEncoderTests(unittest.TestCase):
    def test_forge_krea_encoder_uses_text_processing_wrapper(self):
        from types import SimpleNamespace

        core = SimpleNamespace(layers=[], embed_tokens=object())
        raw_encoder = SimpleNamespace(model=core)
        wrapper = SimpleNamespace(transformer=raw_encoder, process_tokens=lambda *_: None)
        tokenizer = object()
        clip = SimpleNamespace(
            cond_stage_model=SimpleNamespace(qwen3vl_4b=raw_encoder),
            tokenizer=SimpleNamespace(qwen3vl_4b=tokenizer),
        )
        sd_model = SimpleNamespace(
            forge_objects=SimpleNamespace(clip=clip),
            text_processing_engine_qwen=SimpleNamespace(text_encoder=wrapper),
        )

        self.assertEqual(
            current_encoder.get_qwen3vl(sd_model),
            (clip, wrapper, tokenizer, raw_encoder, core),
        )

    def test_forge_qwen_image21_uses_8b_encoder(self):
        from types import SimpleNamespace

        core = SimpleNamespace(layers=[], embed_tokens=object(), lm_head=object())
        raw_encoder = SimpleNamespace(model=core)
        wrapper = SimpleNamespace(transformer=raw_encoder, process_tokens=lambda *_: None)
        tokenizer = object()
        clip = SimpleNamespace(
            cond_stage_model=SimpleNamespace(qwen3vl_8b=raw_encoder),
            tokenizer=SimpleNamespace(qwen3vl_8b=tokenizer),
        )
        sd_model = SimpleNamespace(
            forge_objects=SimpleNamespace(clip=clip),
            text_processing_engine_qwen=SimpleNamespace(text_encoder=wrapper),
        )

        self.assertEqual(
            current_encoder.get_qwen3vl(sd_model),
            (clip, wrapper, tokenizer, raw_encoder, core),
        )

    def test_8b_uses_separate_output_head_and_input_embeddings(self):
        from types import SimpleNamespace

        input_embedding = torch.nn.Embedding(3, 2)
        output_head = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            input_embedding.weight.fill_(2)
            output_head.weight.fill_(3)
        core = SimpleNamespace(embed_tokens=input_embedding, lm_head=output_head)
        hidden = torch.ones((1, 1, 2))

        self.assertEqual(current_encoder._token_logits(core, hidden, None).tolist(), [6.0] * 3)
        torch.testing.assert_close(
            current_encoder._next_token_embedding(core, 1, hidden.device, None),
            torch.full((1, 1, 2), 2.0),
        )

    def test_selected_model_loads_before_encoder_lookup(self):
        from types import SimpleNamespace

        selected = SimpleNamespace(forge_loading_parameters={"checkpoint": "krea", "modules": ["qwen3vl_4b"]}, forge_hash="", sd_model=object())
        loaded = object()
        calls = []

        def reload_model():
            calls.append(True)
            return loaded, True

        self.assertIs(current_encoder.get_selected_sd_model(selected, reload_model), loaded)
        self.assertEqual(calls, [True])

        selected.forge_hash = str(selected.forge_loading_parameters)
        selected.sd_model = loaded
        self.assertIs(current_encoder.get_selected_sd_model(selected, reload_model), loaded)
        self.assertEqual(calls, [True])

    def test_prompt_tokens_keep_one_image_placeholder_per_image(self):
        images = [object(), object()]
        tokens = current_encoder.build_prompt_tokens(Tokenizer(), "Describe it", images)
        self.assertEqual(len([item for item in tokens if isinstance(item, dict)]), 2)
        self.assertTrue(all(item["type"] == "image" for item in tokens if isinstance(item, dict)))

    def test_prompt_tokens_match_qwen_tokenizer(self):
        from transformers import AutoTokenizer

        forge_root = Path(__file__).resolve().parents[3]
        tokenizer_path = forge_root / "backend" / "huggingface" / "krea" / "Krea-2-Raw" / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        prompt = "Describe a red fox"
        actual = current_encoder.build_prompt_tokens(tokenizer, prompt, [])
        expected = tokenizer(
            "<|im_start|>user\nDescribe a red fox<|im_end|>\n<|im_start|>assistant\n"
        )["input_ids"]
        self.assertEqual(actual, expected)

        with_image = current_encoder.build_prompt_tokens(tokenizer, prompt, [object()])
        expected_image = tokenizer(
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            "Describe a red fox<|im_end|>\n<|im_start|>assistant\n"
        )["input_ids"]
        self.assertEqual(
            [151655 if isinstance(item, dict) else item for item in with_image],
            expected_image,
        )

    def test_sampling_respects_top_p_and_greedy_mode(self):
        logits = torch.tensor([10.0, 2.0, 1.0])
        self.assertEqual(current_encoder.sample_token(logits, 0, 1.0), 0)
        with patch.object(torch, "multinomial", return_value=torch.tensor([0])) as sample:
            self.assertEqual(current_encoder.sample_token(logits, 0.7, 0.2), 0)
        self.assertEqual(sample.call_args.args[0].count_nonzero().item(), 1)

    def test_output_projection_applies_fp8_weight_conversion(self):
        forge_root = Path(__file__).resolve().parents[3]
        sys.path[:0] = [str(forge_root), str(forge_root / "modules_forge" / "packages")]
        from types import SimpleNamespace

        raw = torch.tensor([[1.0, 2.0]], dtype=torch.float8_e4m3fn)

        def convert(weight):
            self.assertEqual(weight.dtype, torch.float8_e4m3fn)
            return weight.float() * 2

        embedding = SimpleNamespace(
            weight=raw, bias=None, weight_function=[convert], bias_function=[],
        )
        core = SimpleNamespace(embed_tokens=embedding)
        hidden = torch.ones((1, 1, 2), dtype=torch.float32)
        weight = current_encoder._output_weight(core, hidden)
        torch.testing.assert_close(weight, torch.tensor([[2.0, 4.0]]))
        torch.testing.assert_close(
            current_encoder._token_logits(core, hidden, weight), torch.tensor([6.0]),
        )
        torch.testing.assert_close(
            current_encoder._next_token_embedding(core, 0, hidden.device, weight),
            torch.tensor([[[2.0, 4.0]]]),
        )

    def test_unsupported_current_model_has_clear_error(self):
        class FakeModel:
            forge_objects = type("Objects", (), {"clip": type("Clip", (), {"cond_stage_model": object()})()})()

        with self.assertRaisesRegex(ValueError, "Qwen 3VL 4B and 8B"):
            current_encoder.get_qwen3vl(FakeModel())

    def test_cached_decode_matches_forge_full_forward(self):
        forge_root = Path(__file__).resolve().parents[3]
        sys.path[:0] = [str(forge_root), str(forge_root / "modules_forge" / "packages")]
        from dataclasses import replace
        from backend.nn.llm.llama import Llama2_, Qwen3VL_4BConfig
        from backend.operations import using_forge_operations

        config = replace(
            Qwen3VL_4BConfig(), vocab_size=128, hidden_size=64,
            intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
        )
        config.head_dim = 16
        config.rope_dims = [4, 2, 2]
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            core = Llama2_(config)
        for parameter in core.parameters():
            torch.nn.init.normal_(parameter, std=0.02)
        embeddings = core.embed_tokens(torch.tensor([[5, 6, 7, 8]]))
        expected = core(None, embeds=embeddings)[0][:, -1]

        _, cache = current_encoder._forward_step(
            core, embeddings[:, :3], torch.arange(3).unsqueeze(0),
            [()] * len(core.layers),
        )
        cache = current_encoder._reserve_cache(cache, 1)
        actual, _ = current_encoder._forward_step(
            core, embeddings[:, 3:], torch.tensor([[3]]), cache,
        )
        torch.testing.assert_close(actual[:, -1], expected, rtol=1e-4, atol=1e-4)

    def test_8b_cached_decode_matches_forge_full_forward(self):
        forge_root = Path(__file__).resolve().parents[3]
        sys.path[:0] = [str(forge_root), str(forge_root / "modules_forge" / "packages")]
        from dataclasses import replace
        from backend.nn.llm.llama import Llama2_, Qwen3VL_8BConfig
        from backend.operations import using_forge_operations

        config = replace(
            Qwen3VL_8BConfig(), vocab_size=128, hidden_size=64,
            intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
        )
        config.head_dim = 16
        config.rope_dims = [4, 2, 2]
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            core = Llama2_(config)
        for parameter in core.parameters():
            torch.nn.init.normal_(parameter, std=0.02)
        embeddings = core.embed_tokens(torch.tensor([[5, 6, 7, 8]]))
        full_hidden = core(None, embeds=embeddings)[0]
        expected_logits = core.lm_head(full_hidden[:, -1])[0]

        _, cache = current_encoder._forward_step(
            core, embeddings[:, :3], torch.arange(3).unsqueeze(0),
            [()] * len(core.layers),
        )
        cache = current_encoder._reserve_cache(cache, 1)
        hidden, _ = current_encoder._forward_step(
            core, embeddings[:, 3:], torch.tensor([[3]]), cache,
        )
        actual_logits = current_encoder._token_logits(core, hidden, None)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-4, atol=1e-4)

        image_positions = torch.tensor([[0, 1, 2, 5], [0, 1, 3, 5], [0, 1, 4, 5]])
        expected = core(None, embeds=embeddings, position_ids=image_positions)[0][:, -1]
        _, cache = current_encoder._forward_step(
            core, embeddings[:, :3], image_positions[:, :3],
            [()] * len(core.layers),
        )
        cache = current_encoder._reserve_cache(cache, 1)
        actual, _ = current_encoder._forward_step(
            core, embeddings[:, 3:], torch.tensor([[5]]), cache,
        )
        torch.testing.assert_close(actual[:, -1], expected, rtol=1e-4, atol=1e-4)

        visual_mask = torch.tensor([[False, True, False, False]])
        deepstack = [torch.randn(1, 64) * 0.01 for _ in core.layers]
        expected = core(
            None, embeds=embeddings, position_ids=image_positions,
            visual_pos_masks=visual_mask, deepstack_embeds=deepstack,
        )[0][:, -1]
        _, cache = current_encoder._forward_step(
            core, embeddings[:, :3], image_positions[:, :3],
            [()] * len(core.layers), visual_mask[:, :3], deepstack,
        )
        cache = current_encoder._reserve_cache(cache, 1)
        actual, _ = current_encoder._forward_step(
            core, embeddings[:, 3:], torch.tensor([[5]]), cache,
        )
        torch.testing.assert_close(actual[:, -1], expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
