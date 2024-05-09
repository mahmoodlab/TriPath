import torch.nn as nn
import torch

class BCELogitsCustomLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self,
                x,
                target,
                attn=None,
                coords=None):

        return self.loss(x, target.unsqueeze(dim=0).float())