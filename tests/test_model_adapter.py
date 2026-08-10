import types
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from lib.layerwrapper_gpt2 import WrappedGPT2
from lib.prune_gpt2 import (
    apply_canonical_mask,
    canonical_weight,
    find_layers,
    get_decoder_layers,
    prepare_calibration_input,
    prune_magnitude,
    prune_wanda as prune_wanda_gpt2,
    run_layer,
)
from lib.model_utils import resolve_model_family
from lib.prune import find_layers as find_linear_layers
from lib.prune import get_decoder_layers as get_linear_decoder_layers
from lib.prune import prune_wanda


class TinyGPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([nn.Sequential(Conv1D(6, 4))])
        self.config = types.SimpleNamespace(n_embd=4, use_cache=True)


class ModelAdapterTests(unittest.TestCase):
    def _assert_wanda_prunes_linear_decoder(self, model):
        model.seqlen = 8
        dataloader = [
            (torch.randint(0, model.config.vocab_size, (1, 8)), None)
            for _ in range(2)
        ]
        args = types.SimpleNamespace(
            nsamples=2, seed=0, sparsity_ratio=0.5, use_variant=False
        )
        with patch("lib.prune.get_loaders", return_value=(dataloader, None)):
            prune_wanda(args, model, tokenizer=None, device=torch.device("cpu"))

        for block in get_linear_decoder_layers(model):
            for layer in find_linear_layers(block).values():
                weight = layer.weight.data
                expected = int(weight.shape[1] * args.sparsity_ratio)
                self.assertTrue(torch.all((weight == 0).sum(dim=1) == expected))

    def test_gpt2_layout_is_detected(self):
        model = TinyGPT2()
        self.assertIs(get_decoder_layers(model), model.transformer.h)
        self.assertEqual(list(find_layers(model.transformer.h[0])), ["0"])

    def test_conv1d_canonical_weight_is_output_by_input(self):
        layer = Conv1D(3, 2)
        layer.weight.data.copy_(torch.arange(6).reshape(2, 3))
        expected = torch.tensor([[0, 3], [1, 4], [2, 5]])
        self.assertTrue(torch.equal(canonical_weight(layer), expected))

        mask = torch.zeros_like(expected, dtype=torch.bool)
        mask[1, 0] = True
        apply_canonical_mask(layer, mask)
        self.assertEqual(layer.weight.data[0, 1].item(), 0)
        self.assertEqual(layer.weight.data[1, 0].item(), 3)

    def test_wanda_activation_statistics_match_conv1d_inputs(self):
        layer = Conv1D(3, 2)
        wrapper = WrappedGPT2(layer)
        inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        wrapper.add_batch(inputs, layer(inputs))
        self.assertEqual(tuple(wrapper.scaler_row.shape), (2,))
        self.assertTrue(torch.allclose(wrapper.scaler_row, torch.tensor([10.0, 20.0])))

    def test_magnitude_pruning_uses_canonical_conv1d_weight(self):
        model = TinyGPT2()
        layer = model.transformer.h[0][0]
        layer.weight.data.copy_(torch.arange(1, 25, dtype=torch.float32).reshape(4, 6))
        args = types.SimpleNamespace(sparsity_ratio=0.5)
        prune_magnitude(args, model, tokenizer=None, prune_n=0, prune_m=0)
        self.assertEqual(torch.count_nonzero(layer.weight).item(), 11)

    def test_real_gpt2_block_inputs_can_be_captured_and_replayed(self):
        config = GPT2Config(
            n_layer=2, n_head=2, n_embd=8, n_positions=8, n_ctx=8, vocab_size=32
        )
        model = GPT2LMHeadModel(config)
        model.seqlen = 8
        dataloader = [(torch.randint(0, 32, (1, 8)), None) for _ in range(2)]
        inps, outs, layer_args, layer_kwargs = prepare_calibration_input(
            model, dataloader, torch.device("cpu"), nsamples=2
        )
        self.assertEqual(tuple(inps.shape), (2, 8, 8))
        self.assertEqual(tuple(outs.shape), (2, 8, 8))
        self.assertGreater(len(layer_args) + len(layer_kwargs), 0)
        replayed = run_layer(
            model.transformer.h[0], inps[0].unsqueeze(0), layer_args, layer_kwargs
        )
        self.assertEqual(tuple(replayed.shape), (1, 8, 8))

    def test_wanda_prunes_real_gpt2_blocks_per_output(self):
        config = GPT2Config(
            n_layer=2, n_head=2, n_embd=8, n_positions=8, n_ctx=8, vocab_size=32
        )
        model = GPT2LMHeadModel(config)
        model.seqlen = 8
        dataloader = [(torch.randint(0, 32, (1, 8)), None) for _ in range(2)]
        args = types.SimpleNamespace(
            nsamples=2, seed=0, sparsity_ratio=0.5, use_variant=False
        )
        with patch("lib.prune_gpt2.get_loaders", return_value=(dataloader, None)):
            prune_wanda_gpt2(args, model, tokenizer=None, device=torch.device("cpu"))

        for block in model.transformer.h:
            for layer in find_layers(block).values():
                weight = canonical_weight(layer)
                expected = int(weight.shape[1] * args.sparsity_ratio)
                self.assertTrue(torch.all((weight == 0).sum(dim=1) == expected))

    def test_wanda_supports_llama3_mistral_and_qwen2_architectures(self):
        common = dict(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )
        cases = [
            ("llama", LlamaForCausalLM(LlamaConfig(**common))),
            ("mistral", MistralForCausalLM(MistralConfig(**common))),
            ("qwen2", Qwen2ForCausalLM(Qwen2Config(**common))),
        ]
        for expected_family, model in cases:
            with self.subTest(model_family=expected_family):
                self.assertEqual(resolve_model_family(model), expected_family)
                self._assert_wanda_prunes_linear_decoder(model)


if __name__ == "__main__":
    unittest.main()
