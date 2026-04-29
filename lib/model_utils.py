import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SUPPORTED_MODEL_FAMILIES = {"llama", "mistral", "qwen2", "qwen3"}


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
        "This script currently supports Llama, Mistral, and Qwen decoder-only checkpoints."
    )


def get_model_seqlen(model, requested_seqlen=None):
    if requested_seqlen is not None:
        return requested_seqlen

    for attr in ("max_position_embeddings", "n_positions", "seq_length"):
        seqlen = getattr(model.config, attr, None)
        if seqlen is not None:
            return seqlen

    return 2048


def load_model(
    model_name,
    cache_dir="llm_weights",
    dtype=torch.float16,
    device="cuda",
    seqlen=None,
    trust_remote_code=False,
):
    load_kwargs = {
        "cache_dir": cache_dir,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
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
    model.seqlen = get_model_seqlen(model, seqlen)
    return model


def load_tokenizer(
    model_name,
    cache_dir="llm_weights",
    use_fast=False,
    trust_remote_code=False,
):
    tokenizer_kwargs = {
        "cache_dir": cache_dir,
        "use_fast": use_fast,
        "trust_remote_code": trust_remote_code,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    except ValueError:
        if use_fast:
            raise
        tokenizer_kwargs["use_fast"] = True
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

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
