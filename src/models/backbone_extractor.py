import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.models.vit import VisionTransformer, _load_pretrained_weights


BACKBONE_CHOICES = [
    "mobilenetv2", "mobilenetv3_small", "mobilenetv3_large",
    "resnet34", "resnet50", "resnet101", "densenet121",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "efficientnet_v2_s", "convnext_tiny",
    "squeezenet1_0", "squeezenet1_1", "vgg16",
    "vit_b16", "vit_b32",
]


class BackboneExtractor(nn.Module):
    def __init__(self, backbone, extraction_type, feature_dim, is_2d=True):
        super().__init__()
        self.backbone = backbone
        self.extraction_type = extraction_type
        self.feature_dim = feature_dim
        self.is_2d = is_2d

    def forward(self, x):
        if self.extraction_type == "simple":
            return self.backbone.features(x)
        if self.extraction_type == "simple_relu":
            return F.relu(self.backbone.features(x), inplace=True)
        if self.extraction_type == "resnet":
            return self._extract_resnet(x)
        if self.extraction_type == "vgg":
            return self.backbone.features(x)
        if self.extraction_type == "vit":
            return self._extract_vit(x)

    def _extract_resnet(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def _extract_vit(self, x):
        B = x.shape[0]
        x = self.backbone.patch_embed(x)
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.backbone.positions
        x = self.backbone.pos_dropout(x)
        x = self.backbone.encoder(x)
        x = self.backbone.norm(x)
        return x[:, 0]


def create_backbone(name, pretrained=True):
    if name not in BACKBONE_CHOICES:
        raise ValueError(f"Unknown backbone '{name}'. Available: {BACKBONE_CHOICES}")

    if name == "mobilenetv2":
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        backbone.classifier = nn.Identity()
        return BackboneExtractor(backbone, "simple", 1280, True)

    if name == "mobilenetv3_small":
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        return BackboneExtractor(backbone, "simple", in_features, True)

    if name == "mobilenetv3_large":
        backbone = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        )
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        return BackboneExtractor(backbone, "simple", in_features, True)

    if name == "densenet121":
        backbone = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT if pretrained else None
        )
        feature_dim = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        return BackboneExtractor(backbone, "simple_relu", feature_dim, True)

    if name == "resnet34":
        backbone = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return BackboneExtractor(backbone, "resnet", feature_dim, True)

    if name == "resnet50":
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return BackboneExtractor(backbone, "resnet", feature_dim, True)

    if name == "resnet101":
        backbone = models.resnet101(
            weights=models.ResNet101_Weights.DEFAULT if pretrained else None
        )
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return BackboneExtractor(backbone, "resnet", feature_dim, True)

    if name == "efficientnet_b0":
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        return BackboneExtractor(backbone, "simple", feature_dim, True)

    if name == "efficientnet_b1":
        backbone = models.efficientnet_b1(
            weights=models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
        )
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        return BackboneExtractor(backbone, "simple", feature_dim, True)

    if name == "efficientnet_b2":
        backbone = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        )
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        return BackboneExtractor(backbone, "simple", feature_dim, True)

    if name == "efficientnet_v2_s":
        backbone = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        )
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        return BackboneExtractor(backbone, "simple", feature_dim, True)

    if name == "convnext_tiny":
        backbone = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        )
        feature_dim = backbone.classifier[-1].in_features
        backbone.classifier = nn.Identity()
        return BackboneExtractor(backbone, "simple", feature_dim, True)

    if name == "squeezenet1_0":
        backbone = models.squeezenet1_0(
            weights=models.SqueezeNet1_0_Weights.DEFAULT if pretrained else None
        )
        backbone.classifier = nn.Identity()
        return BackboneExtractor(backbone, "simple", 512, True)

    if name == "squeezenet1_1":
        backbone = models.squeezenet1_1(
            weights=models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None
        )
        backbone.classifier = nn.Identity()
        return BackboneExtractor(backbone, "simple", 512, True)

    if name == "vgg16":
        backbone = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT if pretrained else None
        )
        backbone.classifier = nn.Identity()
        return BackboneExtractor(backbone, "vgg", 512, True)

    if name == "vit_b16":
        model = VisionTransformer(
            img_size=224, patch_size=16, in_channels=3,
            num_classes=2, embed_dim=768, depth=12,
            num_heads=12, mlp_dim=3072, dropout=0.0,
        )
        if pretrained:
            _load_pretrained_weights(model, models.ViT_B_16_Weights, 16, embed_dim=768)
        return BackboneExtractor(model, "vit", 768, False)

    if name == "vit_b32":
        model = VisionTransformer(
            img_size=224, patch_size=32, in_channels=3,
            num_classes=2, embed_dim=768, depth=12,
            num_heads=12, mlp_dim=3072, dropout=0.0,
        )
        if pretrained:
            _load_pretrained_weights(model, models.ViT_B_32_Weights, 32, embed_dim=768)
        return BackboneExtractor(model, "vit", 768, False)
