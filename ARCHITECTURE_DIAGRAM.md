# Attention-Guided Five-Expert Lesion MoE

## Inference architecture

```mermaid
flowchart LR
    I["Input lesion image<br/>256 x 256"]
    B["One shared CNN backbone<br/>ConvNeXt-Tiny"]
    E["Early feature map"]
    M["Intermediate feature map"]
    D["Deep feature map"]
    RGB["Recovered RGB image<br/>range 0 to 1"]

    I --> B
    I --> RGB
    B --> E
    B --> M
    B --> D

    E --> T["Texture expert<br/>local + dilated filters<br/>ECA attention"]
    M --> S["Morphology expert<br/>learned + Sobel cues<br/>CBAM attention"]
    D --> SEM["Semantic expert<br/>deep features<br/>channel gate"]
    RGB --> C["Color expert<br/>chromatic channels<br/>SE attention"]
    E --> BD["Boundary expert<br/>gradient + symmetry cues<br/>spatial attention"]

    T --> A["Align feature maps<br/>to intermediate resolution"]
    S --> A
    SEM --> A
    C --> A
    BD --> A

    A --> P["Global average pooling<br/>five 128-D descriptors"]
    P --> H["Five independent heads<br/>LayerNorm + Dropout + Linear"]
    H --> Z["Complete expert logits<br/>zT, zM, zS, zC, zB"]

    P --> R["Disagreement-aware router<br/>descriptor + global context<br/>cosine/absolute disagreement<br/>expert identity"]
    R --> W["Soft sample-wise weights<br/>wT, wM, wS, wC, wB"]

    Z --> F["Weighted logit fusion<br/>z = sum wi zi"]
    W --> F
    F --> O["Softmax<br/>final class probabilities"]

    classDef input fill:#EAF2FF,stroke:#3973B7,color:#111;
    classDef backbone fill:#EAF7ED,stroke:#3A7D44,color:#111;
    classDef expert fill:#FFF3DF,stroke:#B67416,color:#111;
    classDef router fill:#F3EAFF,stroke:#7550A5,color:#111;
    classDef output fill:#FFE9ED,stroke:#A83D52,color:#111;
    class I,RGB input;
    class B,E,M,D backbone;
    class T,S,SEM,C,BD,A,P,H,Z expert;
    class R,W router;
    class F,O output;
```

## Expert design

| Expert | Input | Specialized processing | Attention | Prediction |
|---|---|---|---|---|
| Texture | Early feature map | Depthwise local and dilated convolutions | ECA channel attention | Complete class logits `zT` |
| Morphology | Intermediate feature map | Learned features fused with Sobel gradient magnitude | CBAM channel-spatial attention | Complete class logits `zM` |
| Semantic | Deep feature map | Projected high-level semantic representation | Learned channel gate | Complete class logits `zS` |
| Color | Recovered RGB image | RGB, opponent-color differences, and color spread | SE channel attention | Complete class logits `zC` |
| Boundary | Early feature map | Gradient magnitude and horizontal/vertical symmetry differences | Spatial attention | Complete class logits `zB` |

All expert maps are projected to 128 channels and aligned to the intermediate
feature resolution. Each expert has its own classifier. Consequently, the
semantic path is one routed expert rather than an always-on baseline, and the
model has no separate no-correction route or additive delta-logit path.

## Training objective and model selection

```mermaid
flowchart LR
    Y["Ground-truth label"]
    FINAL["Fused prediction"]
    EX["Five expert predictions"]
    RP["Router probabilities"]
    EMB["Normalized expert embeddings"]

    FINAL --> LF["Primary classification loss"]
    Y --> LF

    EX --> LE["Auxiliary expert loss<br/>mean across five experts"]
    Y --> LE

    RP --> LG["Router-gain supervision<br/>soft targets from per-expert gain"]
    EX --> LG
    Y --> LG

    RP --> LB["Router balance regularizer"]
    EMB --> LD["Expert diversity regularizer"]

    LF --> TOTAL["Joint training objective"]
    LE --> TOTAL
    LG --> TOTAL
    LB --> TOTAL
    LD --> TOTAL

    TOTAL --> EMA["EMA validation evaluation"]
    EMA --> CKPT["Best checkpoint<br/>minimum validation loss"]
    CKPT --> TEST["Train / validation / test reporting"]

    classDef supervision fill:#EAF2FF,stroke:#3973B7,color:#111;
    classDef loss fill:#FFF3DF,stroke:#B67416,color:#111;
    classDef select fill:#EAF7ED,stroke:#3A7D44,color:#111;
    class Y,FINAL,EX,RP,EMB supervision;
    class LF,LE,LG,LB,LD,TOTAL loss;
    class EMA,CKPT,TEST select;
```

During training, low-probability expert dropout prevents permanent dependence
on a single route. Router-gain targets are derived only from training labels;
validation and test inference use only the learned router. The checkpoint used
for final reporting is selected strictly by minimum validation loss.

## Paper notation

For expert `i` in `{texture, morphology, semantic, color, boundary}`:

`Fi = Expert_i(X, Fearly, Fintermediate, Fdeep)`

`pi = GAP(Fi)`

`zi = Head_i(pi)`

The router receives each descriptor, global expert context, pairwise cosine and
absolute disagreement, and a learned expert-identity embedding:

`w = softmax(Router({pi}) / tau)`, with `sum_i wi = 1`.

The deployed prediction is a convex mixture of complete expert logits:

`z = sum_i wi * zi`, followed by `p(y | X) = softmax(z)`.

Suggested figure caption: **Attention-guided lesion-specialized mixture of
experts with one shared CNN backbone. Hierarchical backbone features and the
recovered RGB image feed five complementary experts with expert-specific
attention. A disagreement-aware router assigns sample-dependent weights to
five complete class predictions, which are fused in logit space. The network
is trained jointly and the reported checkpoint is selected by minimum
validation loss.**
