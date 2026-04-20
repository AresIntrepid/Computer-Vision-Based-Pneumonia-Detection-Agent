import torch.nn as nn
from torchvision import models


def get_backbone(name="efficientnet_b3", frozen=False):
    name = name.lower()
    if name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        out_features = model.classifier[1].in_features
        model.classifier = nn.Identity()
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        out_features = model.fc.in_features
        model.fc = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone '{name}'.")
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    return model, out_features


def unfreeze_backbone(backbone):
    for param in backbone.parameters():
        param.requires_grad = True