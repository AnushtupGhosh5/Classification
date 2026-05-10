import torchvision.models as models
import torch.nn as nn


def create_mobilenetv2(num_classes=2, pretrained=True):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model, "classifier"
