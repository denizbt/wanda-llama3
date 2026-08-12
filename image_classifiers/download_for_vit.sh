#!/usr/bin/env bash

set -euo pipefail

# Run from anywhere; all paths are kept inside image_classifiers by default.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="${WEIGHTS_DIR:-${SCRIPT_DIR}/model_weights/vit}"
IMAGENET_DIR="${IMAGENET_DIR:-${SCRIPT_DIR}/data/imagenet-1k}"
VIT_CHECKPOINT="${WEIGHTS_DIR}/jx_vit_base_p16_224-80ecf9dd.pth"
PYTHON_BIN="${PYTHON_BIN:-python}"
MIN_FREE_GB="${MIN_FREE_GB:-170}"
DIAGNOSTIC_EXAMPLES="${DIAGNOSTIC_EXAMPLES:-}"

mkdir -p "${WEIGHTS_DIR}" "${IMAGENET_DIR}"

# Download the checkpoint expected by models/vision_transformer.py.
if [[ ! -f "${VIT_CHECKPOINT}" ]]; then
    wget -O "${VIT_CHECKPOINT}" \
        https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
fi

# ImageNet-1k is gated on Hugging Face. Accept its terms and run `hf auth login` first. Export the labeled train and validation splits to the ImageFolder layout
# expected by main.py. This is resumable, but requires substantial disk space.
#   ${IMAGENET_DIR}/train/<class>/*.JPEG
#   ${IMAGENET_DIR}/val/<class>/*.JPEG
if [[ -z "${DIAGNOSTIC_EXAMPLES}" && ( ! -f "${IMAGENET_DIR}/train/.complete" || ! -f "${IMAGENET_DIR}/val/.complete" ) ]]; then
    available_kb="$(df -Pk "${IMAGENET_DIR}" | awk 'NR == 2 {print $4}')"
    required_kb="$((MIN_FREE_GB * 1024 * 1024))"
    if (( available_kb < required_kb )); then
        available_gb="$((available_kb / 1024 / 1024))"
        printf 'ImageNet preparation needs about %s GiB free, but %s has only %s GiB. Set IMAGENET_DIR to a larger volume.\n' \
            "${MIN_FREE_GB}" "${IMAGENET_DIR}" "${available_gb}" >&2
        exit 1
    fi
fi

prepare_args=("${IMAGENET_DIR}")
if [[ -n "${DIAGNOSTIC_EXAMPLES}" ]]; then
    if [[ ! "${DIAGNOSTIC_EXAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'DIAGNOSTIC_EXAMPLES must be a positive integer.\n' >&2
        exit 1
    fi
    prepare_args+=(--max-examples-per-split "${DIAGNOSTIC_EXAMPLES}")
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_imagenet.py" "${prepare_args[@]}"

printf 'Preparation complete.\nCheckpoint: %s\nImageNet:   %s\n' \
    "${VIT_CHECKPOINT}" "${IMAGENET_DIR}"
