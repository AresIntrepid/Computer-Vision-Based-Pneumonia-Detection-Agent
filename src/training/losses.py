import torch
import torch.nn as nn


def get_weighted_bce_loss(pos_weight, device):
    weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)