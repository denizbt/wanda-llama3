import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SUPPORTED_MODEL_FAMILIES = {"llama"}


def resolve_runtime(device="auto", dtype="auto"):
    if device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = device

    if dtype == "auto":
        resolved_dtype = "float16" if resolved_device == "cuda" else "float32"
    else:
        resolved_dtype = dtype

    return resolved_device, getattr(torch, resolved_dtype)


def resolve_model_family(model, requested_family="auto"):
    if requested_family != "auto":
        family = requested_family.lower()
    else:
        family = getattr(model.config, "model_type", "").lower()

    if family in SUPPORTED_MODEL_FAMILIES:
        return family

    raise NotImplementedError(
        f"Model family '{family or 'unknown'}' is not supported yet. "
        "This script currently supports Llama-family checkpoints (including Llama 3). "
        "Add the new family here before wiring in architecture-specific handling."
    )


def load_model(model_name, cache_dir="llm_weights", dtype=torch.float16, device="cuda"):
    load_kwargs = {
        "cache_dir": cache_dir,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        load_kwargs["device_map"] = "auto"

    # Newer Transformers prefers `dtype`, while older repo-pinned versions expect
    # `torch_dtype`. Try the newer API first and fall back for compatibility.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            **load_kwargs,
        )
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            **load_kwargs,
        )

    if device in {"cpu", "mps"}:
        model.to(device)
    model.seqlen = model.config.max_position_embeddings
    return model


def load_tokenizer(model_name, cache_dir="llm_weights"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_fast=False,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_pruning_device(model, model_name, runtime_device=None):
    if hasattr(model, "hf_device_map"):
        if "lm_head" in model.hf_device_map and any(
            size_tag in model_name.lower() for size_tag in ("30b", "65b", "66b", "70b")
        ):
            return model.hf_device_map["lm_head"]
        if "model.embed_tokens" in model.hf_device_map:
            return model.hf_device_map["model.embed_tokens"]
    if runtime_device in {"cpu", "mps"}:
        return torch.device(runtime_device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_metadata(output_dir, metadata):
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "pruning_summary.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return metadata_path
