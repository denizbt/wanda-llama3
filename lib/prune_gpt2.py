"""GPT-2-specific magnitude and Wanda pruning.

GPT-2 uses Conv1D projections stored as [input, output].  This module keeps
that architecture-specific handling separate from the original LLM pruning
implementation in ``lib.prune``.
"""

import inspect

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from .data import get_loaders
from .layerwrapper_gpt2 import WrappedGPT2


def get_decoder_layers(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Expected a GPT-2 model with blocks at model.transformer.h.")


def find_layers(module, name=""):
    if isinstance(module, Conv1D):
        return {name: module}
    result = {}
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        result.update(find_layers(child, full_name))
    return result


def canonical_weight(layer):
    """Return a GPT-2 Conv1D weight in Wanda's [output, input] orientation."""
    return layer.weight.data.t()


def apply_canonical_mask(layer, mask):
    layer.weight.data[mask.t()] = 0


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def get_layer_kwargs(layer, kwargs):
    signature = inspect.signature(layer.forward)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def run_layer(layer, hidden_states, args, kwargs):
    output = layer(
        hidden_states,
        *move_to_device(args, hidden_states.device),
        **get_layer_kwargs(layer, move_to_device(kwargs, hidden_states.device)),
    )
    return output[0] if isinstance(output, (tuple, list)) else output


def get_pruning_device(model, runtime_device=None):
    device_map = getattr(model, "hf_device_map", {})
    if "transformer.wte" in device_map:
        return device_map["transformer.wte"]
    if runtime_device in {"cpu", "mps"}:
        return torch.device(runtime_device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    zero_count = 0
    parameter_count = 0
    for index, block in enumerate(get_decoder_layers(model)):
        block_zeros = 0
        block_parameters = 0
        for layer in find_layers(block).values():
            weight = canonical_weight(layer)
            block_zeros += (weight == 0).sum().item()
            block_parameters += weight.numel()
        print(f"layer {index} sparsity {block_zeros / block_parameters:.6f}")
        zero_count += block_zeros
        parameter_count += block_parameters
    model.config.use_cache = use_cache
    return zero_count / parameter_count


def prepare_calibration_input(model, dataloader, device, nsamples):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_decoder_layers(model)
    device_map = getattr(model, "hf_device_map", {})
    if "transformer.wte" in device_map:
        device = device_map["transformer.wte"]

    dtype = next(model.parameters()).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.n_embd), dtype=dtype, device=device
    )
    cache = {"i": 0, "args": (), "kwargs": {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, *args, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["args"] = args
            cache["kwargs"] = kwargs
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.config.use_cache = use_cache
    return inps, torch.zeros_like(inps), cache["args"], cache["kwargs"]


def _rowwise_mask(metric, sparsity_ratio, prune_n, prune_m):
    mask = torch.zeros_like(metric, dtype=torch.bool)
    if prune_n:
        for start in range(0, metric.shape[1], prune_m):
            group = metric[:, start:start + prune_m].float()
            count = min(prune_n, group.shape[1])
            mask.scatter_(1, start + torch.topk(group, count, dim=1, largest=False)[1], True)
    else:
        count = int(metric.shape[1] * sparsity_ratio)
        indices = torch.sort(metric, dim=-1, stable=True)[1][:, :count]
        mask.scatter_(1, indices, True)
    return mask


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    for block in get_decoder_layers(model):
        for layer in find_layers(block).values():
            weight = canonical_weight(layer)
            if prune_n:
                mask = _rowwise_mask(weight.abs(), args.sparsity_ratio, prune_n, prune_m)
            else:
                threshold = torch.sort(weight.abs().flatten())[0][
                    int(weight.numel() * args.sparsity_ratio)
                ]
                mask = weight.abs() <= threshold
            apply_canonical_mask(layer, mask)


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    print("loading calibration data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, tokenizer=tokenizer,
    )
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, layer_args, layer_kwargs = prepare_calibration_input(
            model, dataloader, device, args.nsamples
        )

    device_map = getattr(model, "hf_device_map", {})
    for index, block in enumerate(get_decoder_layers(model)):
        key = f"transformer.h.{index}"
        if key in device_map:
            device = device_map[key]
            inps, outs = inps.to(device), outs.to(device)
            layer_args = move_to_device(layer_args, device)
            layer_kwargs = move_to_device(layer_kwargs, device)

        layers = find_layers(block)
        wrapped = {name: WrappedGPT2(layer) for name, layer in layers.items()}
        handles = []
        for name, layer in layers.items():
            handles.append(layer.register_forward_hook(
                lambda _, inp, out, name=name: wrapped[name].add_batch(inp[0].data, out.data)
            ))
        for sample in range(args.nsamples):
            with torch.no_grad():
                outs[sample] = run_layer(
                    block, inps[sample].unsqueeze(0), layer_args, layer_kwargs
                )
        for handle in handles:
            handle.remove()

        for name, layer in layers.items():
            print(f"pruning layer {index} name {name}")
            metric = canonical_weight(layer).abs() * torch.sqrt(
                wrapped[name].scaler_row.reshape(1, -1)
            )
            mask = _rowwise_mask(metric, args.sparsity_ratio, prune_n, prune_m)
            apply_canonical_mask(layer, mask)

        for sample in range(args.nsamples):
            with torch.no_grad():
                outs[sample] = run_layer(
                    block, inps[sample].unsqueeze(0), layer_args, layer_kwargs
                )
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

