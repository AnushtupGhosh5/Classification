import torch
import torch.nn as nn
import torchvision.models as models


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class MLPBlock(nn.Module):
    def __init__(self, embed_dim=768, mlp_dim=3072, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(self.act(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=2,
                 embed_dim=768, depth=12, num_heads=12, mlp_dim=3072, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positions = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positions, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.positions
        x = self.pos_dropout(x)

        x = self.encoder(x)
        x = self.norm(x)

        cls_output = x[:, 0]
        logits = self.classifier(cls_output)
        return logits


def _load_pretrained_weights(model, weight_cls, patch_size, embed_dim):
    ref_model = models.vit_b_16(weights=weight_cls.DEFAULT) if patch_size == 16 \
        else models.vit_b_32(weights=weight_cls.DEFAULT)
    ref_state = ref_model.state_dict()

    new_state = {}
    new_state["patch_embed.proj.weight"] = ref_state["conv_proj.weight"]
    new_state["patch_embed.proj.bias"] = ref_state["conv_proj.bias"]
    new_state["cls_token"] = ref_state["class_token"]
    new_state["positions"] = ref_state["encoder.pos_embedding"]

    for i in range(12):
        ref_prefix = f"encoder.layers.encoder_layer_{i}"
        custom_prefix = f"encoder.{i}"

        new_state[f"{custom_prefix}.norm1.weight"] = ref_state[f"{ref_prefix}.ln_1.weight"]
        new_state[f"{custom_prefix}.norm1.bias"] = ref_state[f"{ref_prefix}.ln_1.bias"]

        in_proj_weight = ref_state[f"{ref_prefix}.self_attention.in_proj_weight"]
        in_proj_bias = ref_state[f"{ref_prefix}.self_attention.in_proj_bias"]
        q_w, k_w, v_w = in_proj_weight.chunk(3, dim=0)
        q_b, k_b, v_b = in_proj_bias.chunk(3, dim=0)
        new_state[f"{custom_prefix}.attn.q_proj.weight"] = q_w
        new_state[f"{custom_prefix}.attn.q_proj.bias"] = q_b
        new_state[f"{custom_prefix}.attn.k_proj.weight"] = k_w
        new_state[f"{custom_prefix}.attn.k_proj.bias"] = k_b
        new_state[f"{custom_prefix}.attn.v_proj.weight"] = v_w
        new_state[f"{custom_prefix}.attn.v_proj.bias"] = v_b

        new_state[f"{custom_prefix}.attn.proj.weight"] = ref_state[f"{ref_prefix}.self_attention.out_proj.weight"]
        new_state[f"{custom_prefix}.attn.proj.bias"] = ref_state[f"{ref_prefix}.self_attention.out_proj.bias"]

        new_state[f"{custom_prefix}.norm2.weight"] = ref_state[f"{ref_prefix}.ln_2.weight"]
        new_state[f"{custom_prefix}.norm2.bias"] = ref_state[f"{ref_prefix}.ln_2.bias"]

        new_state[f"{custom_prefix}.mlp.fc1.weight"] = ref_state[f"{ref_prefix}.mlp.0.weight"]
        new_state[f"{custom_prefix}.mlp.fc1.bias"] = ref_state[f"{ref_prefix}.mlp.0.bias"]
        new_state[f"{custom_prefix}.mlp.fc2.weight"] = ref_state[f"{ref_prefix}.mlp.3.weight"]
        new_state[f"{custom_prefix}.mlp.fc2.bias"] = ref_state[f"{ref_prefix}.mlp.3.bias"]

    new_state["norm.weight"] = ref_state["encoder.ln.weight"]
    new_state["norm.bias"] = ref_state["encoder.ln.bias"]

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        classifier_keys = [k for k in missing if k.startswith("classifier.")]
        other_missing = [k for k in missing if k not in classifier_keys]
        if other_missing:
            print(f"  Warning: unexpected missing keys: {other_missing}")
    if unexpected:
        print(f"  Warning: unexpected keys in pretrained weights: {unexpected}")

    del ref_model, ref_state


def _create_vit(patch_size, weight_cls, num_classes, pretrained):
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, in_channels=3,
        num_classes=num_classes, embed_dim=768, depth=12,
        num_heads=12, mlp_dim=3072, dropout=0.0,
    )
    if pretrained:
        _load_pretrained_weights(model, weight_cls, patch_size, embed_dim=768)
    return model, "classifier"


def create_vit_b16(num_classes=2, pretrained=True, attention=None):
    if attention and attention != "none":
        print(f"  Note: ViT has built-in multi-head self-attention; '{attention}' ignored")
    return _create_vit(16, models.ViT_B_16_Weights, num_classes, pretrained)


def create_vit_b32(num_classes=2, pretrained=True, attention=None):
    if attention and attention != "none":
        print(f"  Note: ViT has built-in multi-head self-attention; '{attention}' ignored")
    return _create_vit(32, models.ViT_B_32_Weights, num_classes, pretrained)
