import argparse
import os
from importlib.metadata import version

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lib.eval import eval_ppl

# NEW METHODS FOR US
from lib.model_utils import (
    get_pruning_device,
    load_model,
    load_tokenizer,
    resolve_model_family,
    resolve_runtime,
    save_metadata,
)
######

from lib.prune import (
    check_sparsity,
    prune_ablate,
    prune_magnitude,
    prune_sparsegpt,
    prune_wanda,
)


print("torch", version("torch"))
print("transformers", version("transformers"))
print("accelerate", version("accelerate"))
print("# of gpus:", torch.cuda.device_count())


def validate_hf_checkpoint(output_dir, verify_model_load=False):
    config = AutoConfig.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    if verify_model_load:
        reloaded_model = AutoModelForCausalLM.from_pretrained(output_dir)
        model_class = type(reloaded_model).__name__
    else:
        model_class = None

    return {
        "model_type": getattr(config, "model_type", None),
        "tokenizer_class": type(tokenizer).__name__,
        "model_class": model_class,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prune and save a Llama-family causal LM checkpoint."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model id i.e. meta-llama/Meta-Llama-3-8B.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the sparse model and tokenizer will be written.",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        default="llama",
        choices=["llama", "mistral"],
        help="Currently only llama is supported",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--sparsity_ratio", type=float, default=0.5)
    parser.add_argument(
        "--sparsity_type",
        type=str,
        default="unstructured",
        choices=["unstructured", "4:8", "2:4"],
    )
    # these are all the ones supported in this repo
    parser.add_argument(
        "--prune_method",
        type=str,
        default="wanda",
        choices=[
            "magnitude",
            "wanda",
            "sparsegpt",
            "ablate_mag_seq",
            "ablate_wanda_seq",
            "ablate_mag_iter",
            "ablate_wanda_iter",
        ],
    )
    parser.add_argument("--cache_dir", type=str, default="llm_weights")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Runtime device. Defaults to cuda; use auto on MacBook or mixed environments.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="Model load dtype. Auto uses float32 on cpu/mps and float16 on cuda.",
    )
    parser.add_argument("--use_variant", action="store_true")
    parser.add_argument(
        "--eval_ppl",
        action="store_true",
        help="Run the repository's WikiText-2 perplexity evaluation before saving metadata.",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        help="Save the model using safetensors when supported by Transformers.",
    )
    parser.add_argument(
        "--verify_hf_load",
        action="store_true",
        help="Reload the saved checkpoint with Hugging Face after saving to verify compatibility.",
    )
    return parser.parse_args()


def get_pruning_pattern(args):
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        if args.sparsity_ratio != 0.5:
            raise ValueError("Structured N:M sparsity requires --sparsity_ratio 0.5.")
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
    return prune_n, prune_m


def run_pruning(args, model, tokenizer, device, prune_n, prune_m):
    if args.sparsity_ratio == 0:
        return

    print("pruning starts")
    if args.prune_method == "wanda":
        prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "magnitude":
        prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "sparsegpt":
        prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    else:
        prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    prune_n, prune_m = get_pruning_pattern(args)
    runtime_device, runtime_dtype = resolve_runtime(args.device, args.dtype)

    print(f"loading llm model {args.model}")
    model = load_model(
        args.model,
        args.cache_dir,
        dtype=runtime_dtype,
        device=runtime_device,
    )
    model.eval()
    resolved_family = resolve_model_family(model, args.model_family)
    tokenizer = load_tokenizer(args.model, args.cache_dir)

    device = get_pruning_device(model, args.model, runtime_device=runtime_device)
    print("use device", device)

    run_pruning(args, model, tokenizer, device, prune_n, prune_m)

    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*" * 30)

    ppl_test = None
    if args.eval_ppl:
        ppl_test = eval_ppl(args, model, tokenizer, device)
        print(f"wikitext perplexity {ppl_test}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(
        args.output_dir,
        safe_serialization=args.safe_serialization,
    )
    tokenizer.save_pretrained(args.output_dir)

    hf_validation = validate_hf_checkpoint(
        args.output_dir,
        verify_model_load=args.verify_hf_load,
    )
    print(
        "validated Hugging Face checkpoint:",
        f"model_type={hf_validation['model_type']}",
        f"tokenizer={hf_validation['tokenizer_class']}",
        *( [f"model={hf_validation['model_class']}"] if hf_validation["model_class"] else [] ),
    )

    metadata = {
        "model": args.model,
        "model_family": resolved_family,
        "prune_method": args.prune_method,
        "requested_sparsity_ratio": args.sparsity_ratio,
        "actual_sparsity_ratio": sparsity_ratio,
        "sparsity_type": args.sparsity_type,
        "nsamples": args.nsamples,
        "seed": args.seed,
        "cache_dir": args.cache_dir,
        "device": str(device),
        "dtype": str(runtime_dtype),
        "use_variant": args.use_variant,
        "eval_ppl": ppl_test,
        "hf_validation": hf_validation,
    }
    metadata_path = save_metadata(args.output_dir, metadata)
    print(f"saved sparse model to {args.output_dir}")
    print(f"saved pruning metadata to {metadata_path}")


if __name__ == "__main__":
    main()
