# Current Paper Results

These are the locked results currently used for the PAD-UFES-20 and ISIC 2017
study. Checkpoints were selected by minimum validation loss. Test sets were
evaluated after selection and were not used for model development.

## PAD-UFES-20

The dataset is split by patient (70% training, 15% validation, 15% test; seed
42), and only images are passed to the models.

### Test-set comparison

| Model | Accuracy | Balanced accuracy | Macro F1 | Macro ROC-AUC |
|---|---:|---:|---:|---:|
| ConvNeXt-Tiny | **0.7739** | 0.6318 | 0.6599 | **0.9618** |
| EfficientNet-B0 | 0.7507 | 0.6623 | 0.6711 | 0.9485 |
| EfficientNet-B1 | 0.7101 | 0.5987 | 0.5993 | 0.9423 |
| ResNet-50 | 0.6551 | 0.5136 | 0.5179 | 0.9189 |
| MobileNetV2 | 0.6812 | 0.6302 | 0.6118 | 0.9252 |
| DenseNet-121 | 0.7072 | 0.5851 | 0.5840 | 0.9498 |
| ResNet-101 | 0.7130 | 0.5674 | 0.5601 | 0.9315 |
| Proposed five-branch MoE | 0.7710 | **0.6675** | **0.6767** | 0.9587 |

Relative to the matched ConvNeXt-Tiny baseline, the proposed model changes
accuracy by -0.29 percentage points while improving balanced accuracy by 3.57
points and macro F1 by 1.68 points. This is evidence of better class-balanced
performance, not an overall-accuracy improvement on PAD-UFES-20.

### Proposed model by split

| Split | Loss | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Train | 0.0967 | 0.9645 | 0.9568 | 0.9620 |
| Validation | 0.4061 | 0.7543 | 0.6402 | 0.6534 |
| Test | 0.3522 | 0.7710 | 0.6675 | 0.6767 |

## ISIC 2017

Both rows use the predefined ISIC 2017 training, validation, and test sets and
the same ConvNeXt-Tiny backbone family.

### Matched comparison

| Model | Split | Loss | Accuracy | Balanced accuracy | Macro F1 | Macro ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | Validation | 0.3576 | 0.8467 | 0.8045 | 0.8189 | — |
| ConvNeXt-Tiny | Test | 0.4760 | 0.7667 | 0.7184 | 0.7005 | 0.9040 |
| Proposed five-branch MoE | Validation | **0.3404** | **0.8733** | **0.8289** | **0.8469** | — |
| Proposed five-branch MoE | Test | **0.4245** | **0.7983** | **0.7311** | **0.7289** | **0.9177** |

On the ISIC 2017 test set, the proposed model improves accuracy by 3.16
percentage points, balanced accuracy by 1.27 points, macro F1 by 2.84 points,
and macro ROC-AUC by 1.37 points over the matched baseline.

## Interpretation constraints

- Results should not be presented as state of the art without a controlled
  comparison to external methods using the same data and protocol.
- PAD-UFES-20 improvements are concentrated in class-balanced metrics.
- ISIC 2017 shows an improvement in both accuracy and macro metrics.
- A single run does not establish variance; repeated seeds are required for
  confidence intervals or significance testing.

The paper link will be added after publication.
