# Model Support Expansion

This repository adapts the original Wanda codebase to support additional model architectures (Qwen, Mistral etc). We are not the authors of the original project; we have edited the original README to explain how to use our additional code.

## Quick start: ViT and newer Hugging Face LLMs

Install the dependencies first:

```sh
pip install -r requirements.txt
```

### LLaMA 3, Mistral, Qwen, or GPT-2

Run the newer Hugging Face pruning entry point from the repository root. For example:

```sh
python sparsify_llama3_plus.py \
    --model meta-llama/Meta-Llama-3-8B \
    --output_dir out/llama3_8b/unstructured/wanda \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --seqlen 2048
```

The model repository must be accessible to the logged-in Hugging Face account. More model-family examples are included under "Pruning LLaMA 3, Mistral, Qwen, and GPT-2" below.

### ViT on ImageNet

Accept the access conditions for [ImageNet-1k on Hugging Face](https://huggingface.co/datasets/ILSVRC/imagenet-1k), authenticate with `hf auth login`, and run:

```sh
bash image_classifiers/download_for_vit.sh
```

The script only prepares the inputs: it downloads the compatible ViT-B/16 checkpoint and exports the labeled ImageNet-1k training and validation splits into torchvision `ImageFolder` directories. Dataset preparation is automatic and resumable, but ImageNet is large and requires about 170 GiB of free disk space. Set `IMAGENET_DIR` or `WEIGHTS_DIR` to override the default locations. Set `PYTHON_BIN` if the required packages are installed in a non-default Python environment.

After preparation, run pruning and evaluation as a separate step:

```sh
cd image_classifiers
python main.py \
    --model vit_base \
    --data_path data/imagenet-1k \
    --resume model_weights/vit/jx_vit_base_p16_224-80ecf9dd.pth \
    --prune_metric wanda \
    --prune_granularity row \
    --sparsity 0.5 \
    --save_path output/vit_base_wanda/vit_base_wanda_50.pth
```

Note that Appendix A of the WANDA paper finds that per-output pruning (i.e. prune granularity = row) works better for vision models than per-layer.

To prepare a small diagnostic subset before downloading the full dataset, set the number of images to retain from each split:

```sh
DIAGNOSTIC_EXAMPLES=2 \
PYTHON_BIN=/path/to/python \
bash image_classifiers/download_for_vit.sh
```

Then run the diagnostic pruning and evaluation separately:

```sh
cd image_classifiers
python main.py \
    --model vit_base \
    --device cpu \
    --batch_size 2 \
    --num_workers 0 \
    --nsamples 2 \
    --data_path data/imagenet-1k \
    --resume model_weights/vit/jx_vit_base_p16_224-80ecf9dd.pth \
    --prune_metric wanda \
    --prune_granularity row \
    --sparsity 0.5 \
    --save_path output/vit_base_wanda/vit_base_wanda_50.pth
```

Diagnostic preparation skips the full-dataset disk-space check and reuses an already-exported subset. Its accuracy is not meaningful; it is intended only to verify execution.

#### ViT checkpoint details

The script downloads `jx_vit_base_p16_224-80ecf9dd.pth`, a PyTorch state-dictionary checkpoint for ViT-B/16: a base-size Vision Transformer using 16-by-16 image patches at 224-by-224 resolution. It contains the pretrained and ImageNet-1k-fine-tuned tensors in the timm parameter layout expected by `image_classifiers/models/vision_transformer.py` and the `--resume` loader.

This checkpoint represents the same model weights as [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224). The Hugging Face model card explains that its weights were converted from the timm checkpoint. The original code in WANDA repository expects the ".pth" format which is what we use here.

---

## README

# Pruning LLMs by Weights and Activations
Official PyTorch implementation of **Wanda** (Pruning by **W**eights **and a**ctivations), as presented in our paper:

**A Simple and Effective Pruning Approach for Large Language Models** </br>
*Mingjie Sun\*, Zhuang Liu\*, Anna Bair, J. Zico Kolter* (* indicates equal contribution) <br>
Carnegie Mellon University, Meta AI Research and Bosch Center for AI  <br>
[Paper](https://arxiv.org/abs/2306.11695) - [Project page](https://eric-mingjie.github.io/wanda/home.html)

```bibtex
@article{sun2023wanda,
  title={A Simple and Effective Pruning Approach for Large Language Models}, 
  author={Sun, Mingjie and Liu, Zhuang and Bair, Anna and Kolter, J. Zico},
  year={2023},
  journal={arXiv preprint arXiv:2306.11695}
}
```

--- 
<p align="center">
<img src="https://user-images.githubusercontent.com/20168304/273351964-53c3807e-3453-49c5-b855-b620b1026466.png" width=100% height=100% 
class="center">
</p>

Compared to magnitude pruning which removes weights solely based on their magnitudes, our pruning approach **Wanda** removes weights on a *per-output* basis, by the product of weight magnitudes and input activation norms.

## Update
- [x] (9.22.2023) Add [support](https://github.com/locuslab/wanda#pruning-llama-2) for LLaMA-2.
- [x] (9.22.2023) Add [code](https://github.com/locuslab/wanda#ablation-on-obs-weight-update) to reproduce the ablation study on OBS weight update in the paper.
- [x] (10.6.2023) Add new [support](https://github.com/locuslab/wanda#ablation-on-obs-weight-update) for the weight update analysis in the ablation study. Feel free to try it out!
- [x] (10.6.2023) Add [support](https://github.com/locuslab/wanda#zero-shot-evaluation) for zero-shot evaluation.
- [x] (10.20.2023) Add code for pruning OPT models.
- [x] (10.23.2023) Add code for [LoRA fine-tuning](lora_ft).

## Setup
Installation instructions can be found in [INSTALL.md](INSTALL.md).

## Usage
The [scripts](scripts) directory contains all the bash commands to replicate the main results (Table 2) in our paper.

Below is an example command for pruning LLaMA-7B with Wanda, to achieve unstructured 50% sparsity.
```sh
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_7b/unstructured/wanda/ 
```
We provide a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--prune_method`: We have implemented three pruning methods, namely [`magnitude`, `wanda`, `sparsegpt`].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`].
- `--use_variant`: Whether to use the Wanda variant, default is `False`. 
- `--save`: Specifies the directory where the result will be stored.

For structured N:M sparsity, set the argument `--sparsity_type` to "2:4" or "4:8". An illustrative command is provided below:
```sh
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/llama_7b/2-4/wanda/ 
```

### Pruning LLaMA-2
For [LLaMA-2](https://ai.meta.com/llama/) models, replace `--model` with `meta-llama/Llama-2-7b-hf` (take `7b` as an example):
```sh 
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama2_7b/unstructured/wanda/
```
LLaMA-2 results: (LLaMA-2-34b is not released as of 9.22.2023)
|sparsity| ppl              | llama2-7b | llama2-13b | llama2-70b |
|------|------------------|----------|------------|------------|
|-| dense            | 5.12     | 4.57       | 3.12     |
|unstructured 50%| magnitude        | 14.89    | 6.37       | 4.98     |
|unstructured 50%| sparsegpt        | 6.51     | 5.63       | **3.98**  |
|unstructured 50%| wanda            | **6.42** | **5.56**   | **3.98**  |
|4:8| magnitude        | 16.48    | 6.76       | 5.58     |
|4:8| sparsegpt        | 8.12     | 6.60      | 4.59     |
|4:8| wanda            | **7.97** | **6.55**  | **4.47**     |
|2:4| magnitude        | 54.59    | 8.33       | 6.33       |
|2:4| sparsegpt        | **10.17** | 8.32       | 5.40      |
|2:4| wanda            | 11.02    | **8.27**   | **5.16**     |

### Pruning LLaMA 3, Mistral, Qwen, and GPT-2
Use `sparsify_llama3_plus.py` for Hugging Face checkpoints that should be saved back in standard HF format. It supports decoder-only Llama, Mistral, Qwen2/Qwen3, and GPT-2 architecture families, including `mistralai/Mistral-7B-v0.3`, `Qwen/Qwen2.5-7B`, and `openai-community/gpt2`.

```sh
python sparsify_llama3_plus.py \
    --model meta-llama/Meta-Llama-3-8B \
    --output_dir out/llama3_8b/unstructured/wanda \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --seqlen 2048
```

```sh
python sparsify_llama3_plus.py \
    --model mistralai/Mistral-7B-v0.3 \
    --output_dir out/mistral_7b_v03/unstructured/wanda \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --seqlen 2048
```

```sh
python sparsify_llama3_plus.py \
    --model Qwen/Qwen2.5-7B \
    --output_dir out/qwen2_5_7b/unstructured/wanda \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --seqlen 2048
```

If `--model_family` is omitted, the script reads `config.model_type` from the checkpoint. `--seqlen` is optional, but setting it to `2048` matches the common WANDA calibration length and avoids accidentally allocating calibration tensors at the full long-context length of newer checkpoints.

GPT-2 uses Hugging Face `Conv1D` projections whose stored weights are transposed relative to `torch.nn.Linear`. The pruning path converts them to the paper's output-by-input orientation before applying the unchanged WANDA metric and per-output pruning rule. GPT-2 Small has a maximum context length of 1024, so omit `--seqlen` or set it to at most `1024`:

The isolated GPT-2 path currently supports `wanda` and `magnitude`. SparseGPT and the OBS ablation variants remain on the original LLaMA-style path.

```sh
python sparsify_llama3_plus.py \
    --model openai-community/gpt2 \
    --output_dir out/gpt2/unstructured/wanda \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --seqlen 1024
```

### Ablation on OBS weight update
To reproduce the analysis on weight update, we provide our implementation for this ablation. All commands can be found in [this script](scripts/ablate_weight_update.sh).
```sh
for method in ablate_mag_seq ablate_wanda_seq ablate_mag_iter ablate_wanda_iter 
do 
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model decapoda-research/llama-7b-hf \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --prune_method ${method} \
  --save out/llama_7b_ablation/unstructured/
done 
```
Here `ablate_{mag/wanda}_{seq/iter}` means that we use magnitude pruning or wanda to obtain the pruned mask at each layer, then apply weight update procedure with either a sequential style or an iterative style every 128 input channels. For details, please see Section 5 of our [paper](https://arxiv.org/abs/2306.11695).

### Zero-Shot Evaluation
For evaluating zero-shot tasks, we modify the [EleutherAI LM Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) framework so that it could evaluate pruned LLM models. We provide the modified repo in [this link](https://drive.google.com/file/d/1zugbLyGZKsH1L19L9biHLfaGGFnEc7XL/view?usp=sharing). Make sure to download, extract and install this custom `lm_eval` package from the source code.

For reproducibility, we used [commit `df3da98`](https://github.com/EleutherAI/lm-evaluation-harness/tree/df3da98c5405deafd519c2ddca52bb7c3fe36bef) on the main branch. All tasks were evaluated on task version of 0 except for BoolQ, where the task version is 1.

On a high level, the functionality we provide is adding two arguments `pretrained_model` and `tokenizer` in this [function](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/evaluator.py#L17). We can then call this `simple_evaluate` function API from our [codebase](https://github.com/locuslab/wanda/blob/main/lib/eval.py#L148) to evaluate sparse pruned LLMs. To evaluate zero-shot tasks in addition to the WikiText perplexity, pass the `--eval_zero_shot` argument. 

### Speedup Evaluation
The pruning speed for each method is evaluated by the cumulated time spent on pruning (for each layer), without the forward passes.

For inference speedup with structured sparsity, we refer the reader to this [blog post](https://pytorch.org/tutorials/prototype/semi_structured_sparse.html), where  structured sparsity is supported by `PyTorch >= 2.1`. You can switch between the CUTLASS or CuSPARSELt kernel [here](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/sparse/semi_structured.py#L55).

Last, for pruning image classifiers, see directory [image_classifiers](image_classifiers) for details.

## Acknowledgement
This repository is build upon the [SparseGPT](https://github.com/IST-DASLab/sparsegpt) repository.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Questions
Feel free to discuss papers/code with us through issues/emails!

mingjies at cs.cmu.edu  
liuzhuangthu at gmail.com
