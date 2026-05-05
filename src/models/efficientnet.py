import torchvision.models as models
import torch.nn as nn


def create_efficientnet_b0(num_classes=2, pretrained=True):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model, "classifier"
