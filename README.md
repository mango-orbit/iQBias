# PCQR — Prompt-Context-Coupled Query–Value Refinement

Parameter-efficient adaptation of a frozen ViT for deep metric learning, where the
only backbone parameters that may move are the **query and value bias slices** of
the attention blocks, optionally refined by an **image-conditioned query residual**.

Two variants ship here:

| variant | backbone adaptation | needs prompts |
|---|---|---|
| **PCQR** (main) | query+value bias slices **+** low-rank contextual query residual (`context_rank > 0`) | **no** |
| static QV | query+value bias slices only (`context_rank = 0`) | optional |

The contextual residual is zero-initialised at its output, so a run with
`context_rank > 0` starts from the exact static-QV function rather than from a
perturbed backbone.

> **Structural discipline enforced by the code.** `context_rank > 0` requires
> `visual_prompt` and `semantic_proxy` to be **off**; the model raises otherwise.
> The prompt-context configurations are therefore always built on the *static*
> query+value mask, never on the contextual residual.

## Repository layout

```
.
├── run_train.sh              single entry point
├── prepare_config.py         renders an effective config (see "Overrides" below)
├── train.py                  training / evaluation driver
├── options.py                every command-line option
├── architectures/            qv_vit.py (adaptation), vit.py (plain baseline)
├── criteria/                 proxy_anchor.py (one file = one objective)
├── datasets/                 cub200, cars196, stanford_online_products
├── datasampler/  batchminer/  metrics/  utilities/
├── configs/                  one self-contained recipe per experimental arm
└── LICENSE
```

Two files are not shipped: there is **no dataset and no pretrained weight** in
this repository. Both are supplied by the user (see below).

## Environment

Developed and validated against a single-GPU setup. The original recipe used:

```bash
conda create --name dml python=3.8 -y
conda activate dml
conda install -y -c pytorch pytorch=1.12.1=py3.8_cuda11.3_cudnn8.3.2_0 \
    faiss-gpu torchvision cudatoolkit=11.3
conda install -c conda-forge pretrainedmodels joblib -y
pip install "timm>=1.0.20" tqdm regex pyyaml scipy scikit-learn pandas \
            matplotlib openpyxl termcolor
```

Verified versions: Python 3.8.20, torch 1.12.1 (CUDA 11.3), timm 0.9.16,
numpy 1.24.3, scikit-learn 1.3.2. A working `faiss` build is required by the
metric implementation.

## Data

Point `--source-path` at a root that contains **one directory per dataset name** —
the driver appends `/<dataset>` itself, so the layout is:

```
<source_path>/
├── cub200/images/<class>/<image>.jpg
├── cars196/images/<class>/<image>.jpg
└── sop/
    ├── images/...            # or the flat Stanford archive layout
    └── Info_Files/Ebay_{train,test}.txt
```

CUB200-2011, Cars196 and Stanford Online Products keep their standard train/test
splits; no extra preprocessing step is required.

## Pretrained weights

The backbone is taken from `timm`, using the name in `pretrained_model`
(`vit_small_patch16_224_in21k` by default). With an empty `pretrained_checkpoint`
the weights are fetched from the Hugging Face hub on the first run and cached
under `~/.cache/huggingface/`. On a machine without internet access, download a
timm-format checkpoint and pass it explicitly:

```yaml
pretrained_checkpoint: /absolute/path/to/model.safetensors
```

## Quick start

```bash
bash run_train.sh --dataset cub200                 # main variant, GPU 0
bash run_train.sh --dataset cars196 --gpu 1
bash run_train.sh --dataset sop --epochs 120
bash run_train.sh --config configs/pcqr_cub200_arm_vptsp_qv.yaml
bash run_train.sh --dataset cub200 --dry-run       # validate and print only
```

The entry point changes into the repository root before training, because the
objective registry enumerates `./criteria` relative to the working directory.

### Overrides

The driver parses the command line **first** and then applies the config file
wholesale, so a command-line value would always be overwritten by the file.
`prepare_config.py` therefore merges overrides *into* the config and writes the
effective file to `results/_launch/<name>.yaml`:

```
configs/*.yaml   <   --set KEY=VALUE / --epochs N / --name NAME / --source-path P
```

Any key can be overridden, e.g. `--set context_rank=32 --set eval_interval=2`.

## Experimental arms

Every config is self-contained; nothing is inherited.

| config | `bias_mask` | `context_rank` | `visual_prompt` | `semantic_proxy` |
|---|---|---|---|---|
| `pcqr_cub200.yaml` / `pcqr_cars196.yaml` / `pcqr_sop.yaml` | `qv` | 4 | off | off |
| `pcqr_cub200_pcqr_static_qv.yaml` | `qv` | 0 | off | off |
| `pcqr_cub200_baseline_linear.yaml` | `linear` | 0 | off | off |
| `pcqr_cub200_baseline_all_bias_bitfit.yaml` | `bitfit` | 0 | off | off |
| `pcqr_cub200_arm_vpt_v.yaml` | `v` | 0 | **on** | off |
| `pcqr_cub200_arm_vpt_qv.yaml` | `qv` | 0 | **on** | off |
| `pcqr_cub200_arm_vptsp_v.yaml` | `v` | 0 | **on** | **on** |
| `pcqr_cub200_arm_vptsp_qv.yaml` | `qv` | 0 | **on** | **on** |

`bias_mask` selects the trainable slice of the fused `qkv` bias:
`q`, `k`, `v`, `v_out`, `v_fc2`, `qv` (the union of `q` and `v`), plus the
legacy `bitfit` (all biases) and `linear` (no backbone bias) baselines.
`k` is included deliberately: it is exactly null under row-wise softmax, so it is
the negative control.

## Parameter accounting

The fused `qkv` tensor is registered as a whole, so two counts differ and both
are reported at startup:

- **optimizer-visible** backbone bias coordinates — what the optimizer sees;
- **forward-active** coordinates — what can actually change the forward pass
  after the fixed mask and the post-update restoration.

For ViT-S/16 (`L = 12`, `d = 384`):

| variant | forward-active | optimizer-visible |
|---|---|---|
| static QV | 9,216 | 13,824 |
| QV + contextual query (`r = 4`) | 12,336 | 16,944 |

Quote the convention whenever a parameter budget is reported.

## Outputs

```
results/<dataset>/<backbone>/<timestamp>/
├── run_config.yaml            the config actually used
├── effective_parameters.yaml  every option, after config and overrides
├── best_metrics.json          best value per metric and the epoch it occurred
├── experiment_log__vs_defaults__*.txt
└── metrics/                   per-epoch records
```

`results/_launch/` holds the rendered effective configs.

## Reproducibility notes

- `seed` controls the sampler, the head initialisation and the proxy
  initialisation. Single-seed numbers should be reported as point estimates.
- Mixed precision is on by default (`float16`); `--set amp_dtype=bfloat16` is
  available on GPUs with BF16 support.
- cuDNN autotuning is off by default and deterministic convolution selection is
  disabled in both modes, matching the reference runs.
- The evaluation cadence is `eval_start` / `eval_interval`. Note that a short
  smoke run on a config whose `eval_start` is large will train without producing
  a post-training metric — that is the configured behaviour, not a failure.

## Citation and licence

Released under the MIT licence; see `LICENSE`. This implementation builds on the
visual-prompt proxy framework of Ren et al. (ICLR 2024) and on the proxy-anchor
objective of Kim et al., and on the Deep-Metric-Learning-Baselines code base.
Please cite the original works alongside this one.
