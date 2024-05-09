# Assumes single-batch

import torch.nn as nn

class CrossEntropyCustomLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self,
                x,
                target,
                attn=None,
                coords=None):

        return self.loss(x, target)
