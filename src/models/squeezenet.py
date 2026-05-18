import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.models.attention import get_attention_module


class SqueezeNetWithAttention(nn.Module):
    def __init__(self, backbone, attention_module, num_classes):
        super().__init__()
        self.features = backbone.features
        self.attention = attention_module
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def _create_squeezenet(weight_cls, model_fn, num_classes, pretrained, attention):
    backbone = model_fn(weights=weight_cls.DEFAULT if pretrained else None)
    if attention and attention != "none":
        attn_module = get_attention_module(attention, 512)
        model = SqueezeNetWithAttention(backbone, attn_module, num_classes)
        return model, "classifier"
    backbone.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    return backbone, "classifier"


def create_squeezenet1_0(num_classes=2, pretrained=True, attention=None):
    return _create_squeezenet(
        models.SqueezeNet1_0_Weights, models.squeezenet1_0,
        num_classes, pretrained, attention,
    )


def create_squeezenet1_1(num_classes=2, pretrained=True, attention=None):
    return _create_squeezenet(
        models.SqueezeNet1_1_Weights, models.squeezenet1_1,
        num_classes, pretrained, attention,
    )
