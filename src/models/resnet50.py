import torchvision.models as models
import torch.nn as nn


def create_resnet50(num_classes=2, pretrained=True):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Sequential(
    nn.BatchNorm1d(model.fc.in_features),
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, num_classes)
)
    return model, "fc"
