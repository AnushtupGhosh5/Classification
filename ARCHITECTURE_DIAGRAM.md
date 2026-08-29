# Proposed Architecture

The paper uses neutral branch identifiers E1–E5. Descriptions below state the
implemented operations, not guaranteed or exclusive semantic roles.

## Inference path

```mermaid
flowchart LR
    X["Input image"] --> B["Shared ConvNeXt-Tiny"]
    X --> R["Recovered RGB view"]
    B --> L["Early features"]
    B --> M["Intermediate features"]
    B --> H["Deep features"]

    L --> E1["E1<br/>multi-scale convolutions + ECA"]
    M --> E2["E2<br/>learned and gradient cues + CBAM"]
    H --> E3["E3<br/>deep projection + channel gate"]
    R --> E4["E4<br/>chromatic transform + SE"]
    L --> E5["E5<br/>gradient/reflection cues + spatial attention"]

    E1 --> P["Align + global pooling"]
    E2 --> P
    E3 --> P
    E4 --> P
    E5 --> P

    P --> C["Five independent classifiers"]
    P --> D["Disagreement-aware router"]
    C --> Z["Branch logits z1 ... z5"]
    D --> W["Image-dependent weights w1 ... w5"]
    Z --> F["Weighted logit fusion"]
    W --> F
    F --> Y["Final class probabilities"]

    classDef input fill:#eaf2ff,stroke:#3973b7,color:#111;
    classDef backbone fill:#eaf7ed,stroke:#3a7d44,color:#111;
    classDef branch fill:#fff3df,stroke:#b67416,color:#111;
    classDef router fill:#f3eaff,stroke:#7550a5,color:#111;
    classDef output fill:#ffe9ed,stroke:#a83d52,color:#111;
    class X,R input;
    class B,L,M,H backbone;
    class E1,E2,E3,E4,E5,P,C,Z branch;
    class D,W router;
    class F,Y output;
```

E4 receives a recovered, de-normalized RGB view directly from the model input;
it does not come from a ConvNeXt feature layer. The other branches receive
hierarchical maps from the one shared backbone.

## Mathematical summary

Let the shared backbone provide hierarchical maps

\[
(F^{(e)},F^{(m)},F^{(d)})=B(x).
\]

Each branch implements a distinct transformation and produces a descriptor and
complete class prediction:

\[
h_i=\operatorname{GAP}(E_i(x,F^{(e)},F^{(m)},F^{(d)})),\qquad
z_i=H_i(h_i).
\]

The router uses all descriptors and their learned disagreement representation:

\[
w=\operatorname{softmax}(R(h_1,\ldots,h_5)/\tau),
\qquad \sum_{i=1}^{5}w_i=1.
\]

The deployed classifier is a convex mixture in logit space:

\[
z=\sum_{i=1}^{5}w_i z_i,
\qquad p(y\mid x)=\operatorname{softmax}(z).
\]

The primary loss supervises the fused logits. Auxiliary branch supervision and
small balance/diversity regularizers are controlled by command-line arguments,
not dataset-specific constants embedded in the architecture. The reported
checkpoint is selected by minimum validation loss.

## Interpretation figures

Two distinct figures should be reported:

1. Expert-specific Grad-CAM, where branch `Ei` is explained using its own
   logits and a target layer inside that branch.
2. Final-model Grad-CAM, where the fused logits are explained to visualize the
   deployed decision.

These visualizations can demonstrate different evidence patterns, but they do
not prove that a branch corresponds uniquely to a named clinical concept.
