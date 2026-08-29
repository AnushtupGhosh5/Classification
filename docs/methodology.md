# Methodology

## 1. Problem formulation

Let \(\mathcal D=\{(x_n,y_n)\}_{n=1}^{N}\) be a dermoscopic image dataset,
where \(x_n\in\mathbb R^{3\times H\times W}\) and
\(y_n\in\{1,\ldots,C\}\). The goal is to estimate
\(p(y\mid x)\) while retaining complementary information from different
representation depths. The proposed network uses one shared backbone, five
lightweight predictive branches, and an image-dependent router. Thus, its
additional capacity is concentrated in the branch projections and router
rather than five independent CNNs.

> **Figure 1:** End-to-end architecture from the input image through the shared
> hierarchical backbone, E1–E5, disagreement-aware routing, logit fusion, and
> final prediction.

E1–E5 are neutral identifiers. Their implemented operations introduce
different inductive biases, but neither branch names nor attention maps prove
that a branch has learned an exclusive clinical concept.

## 2. Input processing and data protocol

Images are decoded in RGB, resized according to the experiment configuration,
normalized with the ImageNet statistics expected by the pretrained backbone,
and augmented only in the training split. Validation and test transformations
are deterministic. Optional test-time augmentation averages predictions from
the identity, horizontal flip, vertical flip, and combined flip.

PAD-UFES-20 is evaluated as a six-class, image-only task. Metadata identifies
patients, image files, and labels for split construction, but metadata fields
are not classifier inputs. Patients are assigned to disjoint training,
validation, and test subsets with proportions 70/15/15 and seed 42. ISIC 2017
uses its predefined three-class training, validation, and test sets. In both
cases the test split is withheld from model selection and diagnostics.

> **Figure 2:** Dataset protocol, including patient-level PAD-UFES-20 separation
> and the predefined ISIC 2017 splits.

## 3. Shared hierarchical backbone

A ConvNeXt-Tiny backbone \(B\) maps the normalized image to three feature maps:

\[
(F^{e},F^{m},F^{d})=B(x),
\]

where the superscripts denote early, intermediate, and deep taps. Earlier maps
retain greater spatial detail, while deeper maps have larger receptive fields
and stronger abstraction. Tapping one backbone at several depths avoids the
cost and confounding effect of fusing independently pretrained classifiers.

## 4. Five attention-guided branches

Each branch transforms one information source into a feature map with common
channel dimension \(D\). All maps are resized to the intermediate spatial
resolution before pooling.

| Branch | Source | Implemented operations | Attention |
|---|---|---|---|
| E1 | Early backbone map | Local and dilated depthwise transformations | ECA |
| E2 | Intermediate backbone map | Learned representation combined with gradient cues | CBAM |
| E3 | Deep backbone map | Deep projection and residual processing | Learned channel gate |
| E4 | Recovered RGB input | Chromatic and opponent-channel transforms | SE |
| E5 | Early backbone map | Gradient and reflection-difference transforms | Spatial gate |

The recovered RGB view used by E4 is obtained by reversing input
normalization. It is a separate input-space path and does not originate from a
ConvNeXt layer. These operations are architectural priors only: whether a
branch actually relies on a particular type of evidence must be examined
empirically.

For branch \(i\),

\[
A_i=E_i(x,F^{e},F^{m},F^{d}), \qquad
h_i=\operatorname{GAP}(A_i)\in\mathbb R^D.
\]

An independent head gives every branch a complete prediction over all classes:

\[
z_i=W_i\operatorname{Dropout}(\operatorname{LN}(h_i))+b_i,
\qquad z_i\in\mathbb R^C.
\]

Independent logits make branches individually supervisable, measurable,
ablatable, and explainable.

> **Figure 3:** Internal operations and attention placement for E1–E5. Use the
> neutral identifiers prominently; architectural descriptions may appear as
> subtitles.

## 5. Disagreement-aware routing

The router should estimate which branch predictions are useful for a given
image, not merely learn a fixed global preference. Each normalized descriptor
is first projected into a common comparison space:

\[
q_i=\frac{P(h_i)}{\|P(h_i)\|_2}.
\]

For branch \(i\), disagreement with the remaining branches is represented by
pairwise cosine relations and absolute descriptor differences. These terms are
combined with the branch descriptor, the global branch context
\(\bar h=\frac{1}{K}\sum_i h_i\), and a learned branch-identity embedding. A
shared scoring network produces routing scores \(a_i\). Soft routing is

\[
w_i=\frac{\exp(a_i/\tau)}{\sum_{j=1}^{K}\exp(a_j/\tau)},
\qquad w_i\geq0,\quad\sum_iw_i=1,
\]

where \(\tau\) controls routing sharpness. Disagreement only changes the trust
assigned to predictions; it is not concatenated directly into the final class
representation. During training, low-probability branch dropout discourages
permanent dependence on a single route, after which surviving weights are
renormalized.

> **Figure 4:** Router inputs: pooled branch descriptors, global context,
> branch identity, and pairwise disagreement, followed by normalized weights.

## 6. Logit fusion and theoretical motivation

The final logits are a convex mixture of complete branch logits:

\[
z(x)=\sum_{i=1}^{K}w_i(x)z_i(x),
\qquad p(y=c\mid x)=\frac{e^{z_c}}{\sum_{r=1}^{C}e^{z_r}}.
\]

This rule has three useful properties. First, it preserves branch-level
interpretability because each term is a weighted class prediction. Second, the
convex constraint prevents arbitrary amplification solely through router
weights. Third, weighted logit fusion is equivalent to a normalized weighted
geometric pooling of the branch probabilities:

\[
p(c\mid x)\propto\prod_{i=1}^{K}p_i(c\mid x)^{w_i(x)}.
\]

Consequently, a class receives strong fused support when the branches trusted
for that image provide compatible evidence. Unlike averaging fixed ensemble
members, the weighting function changes with the input.

## 7. Training objective

The primary classification loss is evaluated on the fused logits:

\[
\mathcal L_{\mathrm{final}}=\ell(z,y).
\]

Every branch also receives auxiliary supervision,

\[
\mathcal L_{\mathrm{expert}}=
\frac{1}{K}\sum_{i=1}^{K}\ell(z_i,y),
\]

which prevents a weakly routed branch from becoming untrainable. Structural
regularization contains a router-balance term and a representation-diversity
term:

\[
\mathcal L_{\mathrm{balance}}=
K\sum_{i=1}^{K}\left(\bar w_i-\frac1K\right)^2,
\]

\[
\mathcal L_{\mathrm{diversity}}=
\frac{1}{K(K-1)}\sum_{i\ne j}(q_i^\top q_j)^2,
\]

where \(\bar w_i\) is mean branch usage in a minibatch. The complete objective
is

\[
\mathcal L=\mathcal L_{\mathrm{final}}
+\lambda_e\mathcal L_{\mathrm{expert}}
+\lambda_b\mathcal L_{\mathrm{balance}}
+\lambda_d\mathcal L_{\mathrm{diversity}}.
\]

The coefficients are exposed as experiment arguments. They are not tied to a
dataset inside the model definition. An optional gain-supervision term is
available for ablations but should be reported separately when enabled.

## 8. Optimization and checkpoint selection

Models are initialized from ImageNet weights or a matched baseline checkpoint,
optimized with AdamW, and trained with mixed precision. The configured
scheduler, exponential moving average, gradient clipping, augmentation, loss,
and regularization must be reported for each experiment. A checkpoint replaces
the current best model only when validation loss decreases. Early stopping, if
enabled, also monitors validation loss. Accuracy or test performance never
selects the checkpoint.

## 9. Evaluation

The study reports loss, accuracy, macro precision, macro recall, balanced
accuracy, macro F1, macro specificity, and macro one-vs-rest ROC-AUC. Macro and
balanced measures are essential because the class distributions are unequal.
Comparisons use the same split, backbone initialization, preprocessing, and
checkpoint rule.

For interpretation, expert-specific Grad-CAM backpropagates from
`expert_logits[:, i]` to a target convolution within E\(i\). It does not use
the fused logits. A separate final-model Grad-CAM backpropagates from the fused
prediction. Figures begin with the original image, use samples balanced across
classes, show true and predicted labels, and display the router weight beside
each branch map.

> **Figure 5:** Balanced expert-specific Grad-CAM panel for E1–E5.
>
> **Figure 6:** Corresponding final fused-model Grad-CAM panel.

## 10. Complexity and limitations

For PAD-UFES-20, the five-branch ConvNeXt-Tiny model contains 28,294,672
parameters, compared with 27,824,742 for the matched baseline. The additional
469,930 parameters represent a 1.69% increase, since the expensive backbone is
shared.

The architecture does not guarantee branch specialization, causal
interpretability, calibration, or clinical validity. Router weights measure
model preference rather than clinical importance, and Grad-CAM is post-hoc.
Performance must therefore be supported by matched baselines, branch and
attention ablations, router diagnostics, and repeated seeds rather than by
architectural naming alone.
