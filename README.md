# Disagreement-Aware Mixture of Experts for Skin-Lesion Classification

Research code for image-only skin-lesion classification with a single shared
CNN backbone, five attention-guided branches, and sample-dependent logit
fusion. The current manuscript studies **PAD-UFES-20** and **ISIC 2017**.

The manuscript is in preparation. A paper link and formal citation will be
added here after publication.

## What this repository contains

- Standard CNN baselines, including ConvNeXt, EfficientNet, ResNet,
  DenseNet, and MobileNet.
- The proposed `five_expert_moe` model, implemented with one shared backbone.
- Patient-disjoint PAD-UFES-20 splitting and the predefined ISIC 2017 splits.
- Training, checkpoint selection, evaluation, Grad-CAM, and router diagnostics.
- Per-run Markdown reports instead of an accumulating results CSV.

The five branches are reported neutrally as **E1–E5**. Their operations are
designed to expose complementary representations, but the labels do not claim
that any branch has learned a unique clinical concept. See
[ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) for the model flow and
[docs/methodology.md](docs/methodology.md) for the mathematical methodology.
Current paper results are summarized in [RESULTS.md](RESULTS.md).

## Evaluation protocol

| Dataset | Classes | Split protocol | Model input |
|---|---:|---|---|
| PAD-UFES-20 | 6 | Patient-disjoint 70/15/15 split, seed 42 | Images only |
| ISIC 2017 | 3 | Predefined training, validation, and test sets | Images only |

Clinical metadata is not provided to the classifier. During training, the
checkpoint with the **minimum validation loss** is saved. The test split is not
used for training or checkpoint selection and is evaluated only after the
selected checkpoint is loaded. Because both datasets are imbalanced, accuracy
is reported with macro F1, balanced accuracy, and macro ROC-AUC.

## Model overview

One ConvNeXt-Tiny backbone produces early, intermediate, and deep feature maps.
Five lightweight branches transform selected maps or a recovered RGB view,
then produce independent class logits. A disagreement-aware router observes
the branch descriptors and assigns a normalized weight to each branch for each
image. The final logits are

\[
z(x)=\sum_{i=1}^{5}w_i(x)z_i(x),\qquad
w_i(x)\geq0,\quad\sum_iw_i(x)=1.
\]

The PAD-UFES-20 configuration has **28,294,672 parameters**, compared with
27,824,742 for its ConvNeXt-Tiny baseline: an increase of 469,930 parameters
(1.69%).

## Setup

The supported workflow uses Docker and an NVIDIA GPU.

```bash
docker build -t thyroid-classification .
```

Place the datasets under the ignored `data/` directory:

```text
data/
├── PAD-UFES-20_classification/
│   ├── images/
│   └── metadata.csv
└── ISIC_17/
    ├── Train/
    ├── Valid/
    └── Test/
```

The exact ISIC directory names may follow the alternatives recognized by the
dataset loader. PAD metadata is used only to form patient-disjoint splits and
map image labels; metadata fields are not model inputs.

## Running experiments

`runScript.sh` is the editable experiment recipe executed inside the container:

```bash
./run.sh
```

Each run stores its selected checkpoint under
`outputs/models/<dataset>/<run>/` and its figures and report under
`outputs/results/<dataset>/<run>/`. The primary summary is always:

```text
outputs/results/<dataset>/<run>/results.md
```

The current `runScript.sh` contains the matched PAD-UFES-20 DenseNet-121 and
ResNet-101 baseline recipes. Other models and ablations are controlled through
the arguments in `src/main.py`; no dataset-specific result is hardcoded into
the model.

## Repository layout

```text
.
├── src/
│   ├── data/                 # split construction and preprocessing
│   ├── models/               # baselines and proposed architecture
│   ├── main.py               # training/evaluation entry point
│   ├── evaluate.py           # metrics and diagnostics
│   └── gradcam.py            # expert and fused-model Grad-CAM
├── ARCHITECTURE_DIAGRAM.md
├── RESULTS.md
├── run.sh
├── runScript.sh
└── Dockerfile
```

Datasets, checkpoints, plots, and local caches are intentionally excluded from
version control.

## Reproducibility notes

- Report the split protocol, seed, checkpoint rule, and all macro metrics.
- Compare an MoE run only with a baseline trained on the same split and recipe.
- Treat Grad-CAM and router weights as post-hoc evidence, not proof of a fixed
  semantic role for an individual branch.
- Current headline numbers are single-run results unless repeated-run
  statistics are explicitly stated.

## Citation

Manuscript in preparation. The citation and publication link will be added
after publication.

## Disclaimer

This repository is for research only. It is not a medical device and must not
be used for clinical diagnosis or treatment decisions.
