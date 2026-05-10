import torchvision.models as models
import torch.nn as nn


def create_densenet121(num_classes=2, pretrained=True):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model, "classifier"
