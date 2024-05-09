import torch
import torch.nn as nn


class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py

    Parameters
    ----------
    alpha: float
        TODO: document

    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.

    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def check(self, z, c_t):

        return True

    def __call__(self, h, target=None, coords=None, attn=None):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).

        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """
        y_true = target[:, 2].unsqueeze(1)
        c = (~target[:, 1].bool()).int().unsqueeze(1)

        return nll_loss(h=h, y_true=y_true, c=c,
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


def nll_loss(h, y_true, c, alpha=0.0, eps=1e-7, reduction='mean'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).

    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py

    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).

    y_true: (n_batches, 1)
        The true time bin index label.

    c: (n_batches, 1)
        The censoring status indicator.

    alpha: float
        TODO: document

    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.

    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']

    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """

    # make sure these are ints
    y_true = y_true.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h)
    S = torch.cumprod(1 - hazards, dim=1)

    ###########################
    # Previous implementation #
    ###########################
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # hazards[y] = hazards(1)
    # S[1] = S(1)

    # print("Censor: ", c)
    # print("Hazard: ", hazards)
    # print("Surv func: ", S)

    s_prev = torch.gather(S_padded, dim=1, index=y_true).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y_true).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y_true+1).clamp(min=eps)
    # print('s_prev.s_prev', s_prev.shape, s_prev)
    # print('h_this.shape', h_this.shape, h_this)
    # print('s_this.shape', s_this.shape, s_this)
    # print("=-=======================")

    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    # ##########################
    # # Current implementation #
    # ##########################
    # s_this = torch.gather(S, dim=1, index=y_true).clamp(min=eps)
    # h_this = torch.gather(hazards, dim=1, index=y_true).clamp(min=eps)
    #
    # print("survival ", S, y_true, s_this)
    # print("hazard ", hazards, y_true, h_this)
    # print("Censorship ", c)
    # print("***********************")
    # loss = torch.log(s_this) + (1-c) * torch.log(h_this)
    # loss = -loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss
