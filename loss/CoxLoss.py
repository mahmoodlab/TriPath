import torch.nn as nn
import torch
import numpy as np
from itertools import combinations

# class CoxLoss_alternative(nn.Module):
#     """
#     Implements the Cox proportional hazards loss for deep learning models. See Equation (4) of (Katzman et al, 2018) without the L2 term.
#
#     Parameters
#     ----------
#     reduction: str
#         Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum'].
#
#     References
#     ----------
#     Katzman, J.L., Shaham, U., Cloninger, A., Bates, J., Jiang, T. and Kluger, Y., 2018. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. BMC medical research methodology, 18(1), pp.1-12.
#     """
#     def __init__(self, reduction='mean'):
#         super().__init__()
#         assert reduction in ['sum', 'mean']
#         self.reduction = reduction
#
#     def forward(self, z, c_t):
#         """
#         Parameters
#         ----------
#         z: (batch_size, 1)
#             The predicted log risk scores i.e. h(x) in (Katzman et al, 2018).
#
#         c_t: (batch_size, 2)
#             first element: censorship
#             second element: survival time
#         """
#
#         censor = c_t[:, 0].bool()
#         events = ~censor
#         times = c_t[:, 1]
#
#         z = z.rehape(-1)
#         exp_z = torch.exp(z)
#
#         batch_size = z.shape[0]
#
#         ###############################################################
#         # determine risk set for each observation with observed event #
#         ###############################################################
#
#         event_risk_sets = {}  # risk set for everyone with observed event
#         for (idx_a, idx_b) in combinations(range(batch_size), 2):
#             time_a, event_a = times[idx_a], events[idx_a]
#             time_b, event_b = times[idx_b], events[idx_b]
#
#             # event_idx = experienced event and definietly died
#             # before the still_alive_idx
#             event_idx = None
#             still_alive_idx = None
#             if time_a <= time_b and event_a:
#                 event_idx = idx_a
#                 still_alive_idx = idx_b
#
#             elif time_b <= time_a and event_b:
#                 event_idx = idx_b
#                 still_alive_idx = idx_a
#
#             # risk_sets[event_idx] = list of idxs in risk set for event_idx
#             if event_idx is not None:
#                 if event_idx not in event_risk_sets.keys():
#                     event_risk_sets[event_idx] = [still_alive_idx]
#                 else:
#                     event_risk_sets[event_idx].append(still_alive_idx)
#
#         ############################################################
#         # compute loss terms for observations with observed events #
#         ############################################################
#
#         # if there are no comparable pairs then just return zero
#         if len(event_risk_sets) == 0:
#             # TODO: perhaps return None?
#             return torch.zeros(1, requires_grad=True)
#
#         # Compute each term in the sum in Equation (4) of (Katzman et al, 2018)
#         summands = []
#         for event_idx, risk_set in event_risk_sets.items():
#
#             sum_exp_risk_set = torch.sum([exp_z[r_idx] for r_idx in risk_set])
#             summand = z - torch.log(sum_exp_risk_set)   # ANDREW: THIS IS WRONG
#
#             summands.append(summand)
#
#         ##########
#         # Output #
#         ##########
#
#         if self.reduction == 'mean':
#             return -torch.mean(summands)
#         if self.reduction == 'sum':
#             return -torch.sum(summands)


# This may have some issues
# TODO: remove
class CoxLoss(nn.Module):
    """
    Implements the Cox PH loss. Borrowed from Richard's Pathomic Fusion code

    Parameters
    ----------

    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum'].
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        assert reduction in ['sum', 'mean']
        self.reduction = reduction

    def check(self, z, c_t):
        """
        Check whether cox loss is computable, i.e., numOfevents != 0
        """
        censor = c_t[:, 0].bool()
        events = ~censor
        numOfevents = torch.sum(events)

        return numOfevents > 0

    def forward(self, z, c_t):
        """
        Parameters
        ----------
        z: (batch_size, 1)
            The predicted risk scores.

        c_t: (batch_size, 2)
            first element: censorship
            second element: survival time
        """

        hazards = z
        censor = c_t[:, 0].bool()
        events = ~censor
        survtime = c_t[:, 1]
        numOfevents = torch.sum(events)

        batch_size = hazards.shape[0]
        R_mat = np.zeros([batch_size, batch_size], dtype=int)
        for i in range(batch_size):
            for j in range(batch_size):
                R_mat[i, j] = survtime[j] >= survtime[i]

        # convert to torch and put on same device as hazards
        R_mat = hazards.new(R_mat)

        R_mat = R_mat.float()
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)

        summands = theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))
        # (Important) Need to normalize by number of events
        summands = summands * events / numOfevents
        # summands = summands * events

        # print(torch.sum(summands).item(), torch.mean(summands).item(), (torch.sum(summands)/torch.sum(events)).item())

        # print(torch.mean(summands).item())

        if self.reduction == 'mean':
            return -torch.mean(summands)
        if self.reduction == 'sum':
            return -torch.sum(summands)
