import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata



def plot_attention(scores,
                   z_unique_list,
                   patch_size_z,
                   fpath='.',
                   sigma=7,
                   cmap='jet'):

    z_start = z_unique_list[0]
    z_end = z_unique_list[-1] + patch_size_z

    total = z_end - z_start

    attn = np.zeros(total)
    counter = np.zeros(total)

    for idx, (z_level, score) in enumerate(zip(z_unique_list, scores.flatten())):
        counter[z_level - z_start: z_level - z_start + patch_size_z] += 1
        attn[z_level - z_start: z_level - z_start + patch_size_z] += score

    attn = attn / counter
    attn = rankdata(attn, 'average') / len(attn)

    attn = gaussian_filter(attn, sigma=sigma)

    cm = plt.get_cmap(cmap)
    colors = cm(attn)[:, :3]
    fig, ax = plt.subplots(figsize=(6, 1))
    bars = ax.bar(np.arange(total), np.ones(total), width=1)

    for bar, c in zip(bars, colors):
        c = np.expand_dims(np.expand_dims(c, axis=0), axis=0)
        bar.set_zorder(1)
        bar.set_facecolor("none")
        x, y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()

        ax.imshow(c, extent=[x, x + w, y, y + h], aspect="auto", zorder=0, cmap=cmap)

    ax.set_xlim([0, total])
    ax.set_yticks([])
    ax.set_xlabel("Slices")
    plt.title("Attention scores")

    path = os.path.join(fpath, 'attention_inter_slices.png')
    plt.savefig(path, bbox_inches='tight')
