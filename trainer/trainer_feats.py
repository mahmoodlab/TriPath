"""
Done
"""

import torch
import os
from tqdm import tqdm
from utils.eval_utils import BaseStreamEvaler

def train_loop(model,
               loader,
               optimizer,
               loss_func,
               evaler=None,
               evaler_reset=True,
               grad_accum=4,
               scheduler=None,
               device=None):
    """
    Each training epoch

    Args:
    - model (torch.NN.module): attention model
    - grad_accum (int): Number of gradient accumulation steps
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    if evaler is not None and evaler_reset:
        assert isinstance(evaler, BaseStreamEvaler)
        evaler.reset_tracking()

    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(loader)):
        index, data, coords, y_true = batch

        data = data.to(device)
        y_true = y_true.to(device)
        z, attn_dict = model(data, coords)
        loss = loss_func(z,
                         target=y_true,
                         coords=coords,
                         attn=attn_dict['inter'])

        total_loss += loss.item()

        loss = loss / grad_accum

        loss.backward()

        if ((batch_idx + 1) % grad_accum == 0) or ((batch_idx + 1) == len(loader)):
            optimizer.step()
            optimizer.zero_grad()

        if evaler is not None:
            evaler.log(index=index, z=z, y_true=y_true)

    total_loss /= len(loader.dataset)

    if scheduler is not None:
        scheduler.step()
        print("\nLearning rate: ", scheduler.get_last_lr())

    print("\nLoss: ", total_loss)
    if evaler is not None:
        metrics = evaler.get_metrics()
        print("metrcs: ", metrics)
    else:
        metrics = None

    return total_loss, metrics


def eval_loop(model,
              loader,
              loss_func,
              evaler=None,
              evaler_reset=True,
              device=None,
              save_params=None,
              verbose=False):
    """
    Evaluation loop

    Args:
    - evaler_reset (boolean):
        If true, evaler gets the metrics saves the newly tracked data
        If false, evaler just logs the prediction without getting metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    if evaler is not None and evaler_reset:
        assert isinstance(evaler, BaseStreamEvaler)
        evaler.reset_tracking()

    total_loss = 0
    attn_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            index, data, coords, y_true = batch

            data = data.to(device)
            y_true = y_true.to(device)

            z, attn_dict = model(data, coords)

            loss = loss_func(z, y_true)
            total_loss += loss.item()

            attn_list.append(attn_dict['inter'].squeeze().detach().cpu().numpy())
            # print(np.min(attn_dict['intra'].cpu().numpy()), np.max(attn_dict['intra'].cpu().numpy()))
            if evaler is not None:
                evaler.log(index=index, z=z, y_true=y_true)

            # print(z)

    total_loss /= len(loader.dataset)
    print("\nLoss: ", total_loss)

    if evaler is not None:
        metrics = evaler.get_metrics()
        print("metrcs: ", metrics)

        if save_params is not None:
            path = save_params['path']
            epoch = save_params['epoch']
            split = save_params['split']

            evaler.save_tracked_data(os.path.join(path,
                                                  '{}_epoch_{}.pkl'.format(split, epoch)))
    else:
        metrics = None

    info = {'attn': attn_list}

    return total_loss, metrics, info


