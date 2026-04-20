import torch.nn as nn


class BinaryClassificationHead(nn.Module):
    def __init__(self, in_features, dropout_p=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.head(x).squeeze(1)