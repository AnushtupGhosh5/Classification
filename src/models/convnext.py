import torch.nn as nn
import torchvision.models as models


def create_convnext_tiny(num_classes=2, pretrained=True, attention=None):
    """ImageNet-pretrained ConvNeXt-Tiny with a configurable classifier head."""
    if attention and attention != "none":
        raise ValueError("ConvNeXt-Tiny does not currently support an extra attention block")
    model = models.convnext_tiny(
        weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None,
    )
    in_features = model.classifier[-1].in_features
    # Keep LayerNorm and flatten from torchvision, adding explicit head dropout
    # so --classifier-dropout works consistently with EfficientNet.
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes),
    )
    return model, "classifier"
