import os
import numpy as np
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator


def plot_KM(surv_dict, event_dict, title=None, fname=None):

    plt.figure()

    for key, surv_list in surv_dict.items():
        event_list = event_dict[key]
        time_step, surv_step = kaplan_meier_estimator(event_list, surv_list)

        time_step = np.concatenate(([0], time_step))
        surv_step = np.concatenate(([1.0], surv_step))

        label = '{} (n={})'.format(key, len(event_list))
        plt.step(time_step, surv_step, where="post", label=label)

    if title is not None:
        plt.title("{}".format(title))

    plt.ylabel("est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.legend(loc="best")
    plt.xlim(0, 2000)

    plt.savefig(fname, bbox_inches='tight')
