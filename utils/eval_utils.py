"""
Evaluation class to be plugged in each iteration of training
"""

import numpy as np
from scipy.special import softmax, expit
import torch
import pickle

from sklearn.metrics import roc_auc_score,\
    accuracy_score, balanced_accuracy_score, f1_score
from sksurv.metrics import integrated_brier_score, concordance_index_censored
from sksurv.util import Surv

from loss.CoxLoss import CoxLoss


class BaseStreamEvaler:
    """
    Base class for evaulating supervised learning metrics.
    """
    def reset_tracking(self):
        """
        Resets the tracked data
        """
        self.tracking_ = {'index': [], 'z': [], 'y_true': []}

    def log(self, index, z, y_true):
        assert z.ndim == 2
        # move to numpy
        if torch.is_tensor(z):
            index = index.detach().cpu().numpy()
            z = z.detach().cpu().numpy()
            y_true = y_true.detach().cpu().numpy()

        self.tracking_['index'].append(index)
        self.tracking_['z'].append(z)
        self.tracking_['y_true'].append(y_true)

    def get_tracked_data(self):
        """
        Gets the tracked predictions and true response data.

        Args:
        - None

        Returns:
        - z (np array): [n_samples_tracked, n_out] The predictions for each sample tracked thus far.
        - y_true: (np.array): [n_samples_tracked, n_response] The responses for each sample tracked thus far.
        """
        index = np.concatenate(self.tracking_['index'])
        z = np.concatenate(self.tracking_['z'])
        y_true = np.concatenate(self.tracking_['y_true'])

        z = safe_to_vec(z)
        y_true = safe_to_vec(y_true)

        return index, z, y_true

    def save_tracked_data(self, fpath):
        """
        Saves the tracked z and y_true data to disk.

        Parameters
        ----------
        fpath: str
            File path to save.
        """
        index, z, y_true = self.get_tracked_data()
        to_save = {'index': index, 'z': z, 'y_true': y_true}

        with open(fpath, 'wb') as f:
            pickle.dump(to_save, f)

    def get_preds(self):
        """
        Computes prediction to be used for getting metrics
        """
        raise NotImplementedError("Subclass should overwrite.")

    def get_metrics(self):
        """
        Gets a variety of metrics after all the samples have been logged.

        Output
        ------
        metrics: dict of floats
        """
        raise NotImplementedError("Subclass should overwrite.")


def safe_to_vec(a):
    """
    Ensures a numpy array that should be a vector is always a vector
    """
    a = np.array(a)
    if a.ndim == 2 and a.shape[1] == 1:
        return a.reshape(-1)
    else:
        return a


class ClfEvaler(BaseStreamEvaler):
    """
    Evaluates classification metrics when the predictions are computed in batches.

    Parameters
    ----------
    class_names: None, list of str
        (Optional) The names of each class.
    """
    def __init__(self, class_names=None, loss='cross'):
        self.class_names = class_names
        self.loss = loss
        self.reset_tracking()

    def get_preds(self):
        idx, z, y_true = self.get_tracked_data()
        y_pred, prob_pred = pred_clf(z, self.loss)

        return y_true, y_pred, prob_pred

    def get_metrics(self):
        y_true, y_pred, prob_pred = self.get_preds()

        clf_report = {}
        clf_report['acc'] = accuracy_score(y_true=y_true,
                                           y_pred=y_pred)
        clf_report['bal_acc'] = balanced_accuracy_score(y_true=y_true,
                                                        y_pred=y_pred)

        clf_report['f1'] = f1_score(y_true=y_true,
                                    y_pred=y_pred,
                                    average='macro')

        if len(self.class_names) == 2:
            y_score = prob_pred[:, 1]
        else:
            y_score = prob_pred

        clf_report['auc'] = -1.0
        if len(y_true) > 1:
            clf_report['auc'] = roc_auc_score(y_true=y_true,
                                            y_score=y_score,
                                            average='macro',
                                            multi_class='ovr')

        # prediction counts for each class
        if self.class_names is not None:
            n_classes = len(self.class_names)
        else:
            # try to guess
            n_classes = max(max(y_pred), max(y_true)) + 1

        return clf_report

############
# Survival #
############
def estimate_ibs(event_status, surv_times, surv_func, time_bins, eps = 0.001):
    """
    Compute integrated Brier score (Currently only computing once for all dataset)

    Args:
    - event_status: array of booloean (n_samples, )
        Event status
    - surv_times: array of floats (n_samples, )
        survival days for patients
    - surv_func: array of floats (n_samples, n_time_bins)
        Estimated survival function at evluation time points
    - time_bins: array of floats
        Time points at which to evaluate Brier score
    """

    assert len(time_bins) == surv_func.shape[-1], "Number of time bins do not match"

    survival_train = Surv.from_arrays(event=event_status.astype(bool), time=surv_times)
    survival_test = Surv.from_arrays(event=event_status.astype(bool), time=surv_times)

    time_to_eval_at = []
    for idx, t in enumerate(time_bins):
        if idx == 0:
            time_to_eval_at.append(t + eps)
        elif idx == len(time_bins) - 1:
            time_to_eval_at.append(t - eps)
        else:
            time_to_eval_at.append(t)

    try:
        ibs = integrated_brier_score(survival_train=survival_train,
                                     survival_test=survival_test,
                                     estimate=surv_func,
                                     times=time_to_eval_at)
    except:
        print('An error occurred while computing IBS')
        ibs = -1

    return ibs

def get_perm_c_index_quantile(event, time, n_perm=1000, q=0.95):
    """
    Gets the qth quantile from the permutation distribution of the c-index.

    Parameters
    ----------
    event : array-like, shape = (n_samples,)
        Boolean array denotes whether an event occurred.

    time : array-like, shape = (n_samples,)
        Array containing the time of an event or time of censoring.

    n_perm: int
        Number of permutation samples to draw.

    q : array_like of float
        Quantile or sequence of quantiles to compute, which must be between 0 and 1 inclusive.

    Output
    ------
    quantiles: float
        The qth quantile of the permutation distribution.
    """

    perm_samples = []
    random_estimate = np.arange(len(time))
    for _ in range(n_perm):

        # randomly permuted estimate!
        random_estimate = np.random.permutation(random_estimate)

        ci_perm = concordance_index_censored(event_indicator=event,
                                             event_time=time,
                                             estimate=random_estimate)[0]

        perm_samples.append(ci_perm)

    return np.quantile(a=perm_samples, q=q)

###################
# Survival evaler #
###################
class SurvMetricsMixin:
    """
    Mixin for computing survival metrics.

    Args:
    - tied_tol: float
        The tolerance value for considering ties.  See sksurv.metrics. concordance_index_censored.
    - train_times: None, array-like of floats, (n_samples_train, )
        (Optional) The training data survival times. Used for cumulative_dynamic_auc. If not provided this will not be computed.
    - train_events: None, array-like of bools, (n_samples_train, )
        The training data event indicators. Used for cumulative_dynamic_auc. If not provided this will not be computed.
    """

    def compute_surv_metrics(self, pred_risk_score, events, times):
        """
        Gets the c index. Safely handles NaNs.

        Parameters
        ----------
        pred_risk_score: array-like, (n_samples, )
            The predictted risk score.

        events: array-like of bools, (n_samples, )
            The event indicators

        times: array-like,e (n_samples, )
            The observed survival times

        Output
        ------
        out: dict
            Various survival prediction metrics. Currently only contains c-index
        """
        out = {}

        try:
            out['c_index'] = \
                concordance_index_censored(event_indicator=events,
                                           event_time=times,
                                           estimate=pred_risk_score,
                                           tied_tol=self.tied_tol)[0]

        except Exception as e:
            out['c_index'] = np.nan

        return out


class DiscreteSurvivalEvaler(BaseStreamEvaler, SurvMetricsMixin):
    """
    Evaulation object handling batch predictions for discrete survival models. Computes the concordance index when the predictions are computed in batches.

    See https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html.

    Parameters
    ----------n
    tied_tol: float
        The tolerance value for considering ties.  See sksurv.metrics. concordance_index_censored.

    References
    ----------
    Pölsterl, S., 2020. scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn. J. Mach. Learn. Res., 21(212), pp.1-6.

    Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L., Rosati, R.A, "Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors", Statistics in Medicine, 15(4), 361-87, 1996.
    """
    def __init__(self, tied_tol=1e-8):
        self.tied_tol = tied_tol
        self.reset_tracking()

    def get_preds(self):
        _, z, y_true = self.get_tracked_data()
        pred_risk, surv_func = pred_discr_surv(z)

        return y_true, pred_risk, surv_func

    def get_metrics(self):
        _, z, y_true = self.get_tracked_data()

        survival_time_true = y_true[:, 0]
        event_indicator = y_true[:, 1].astype(bool)

        # compute prpedicted risk
        pred_risk, surv_func = pred_discr_surv(z)

        out = self.compute_surv_metrics(pred_risk_score=pred_risk,
                                        events=event_indicator,
                                        times=survival_time_true)

        return out


class CoxSurvivalEvaler(BaseStreamEvaler, SurvMetricsMixin):
    """
    Evaulation object handling batch predictions for cox survival models. Computes the concordance index and the cox loss function.

    See https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html.

    Parameters
    ----------
    tied_tol: float
        The tolerance value for considering ties.  See sksurv.metrics. concordance_index_censored.

    References
    ----------
    Pölsterl, S., 2020. scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn. J. Mach. Learn. Res., 21(212), pp.1-6.

    Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L., Rosati, R.A, "Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors", Statistics in Medicine, 15(4), 361-87, 1996.
    """
    def __init__(self, tied_tol=1e-8, train_times=None, train_events=None):
        self.tied_tol = tied_tol
        self.reset_tracking()

        self.train_times = train_times
        self.train_events = train_events
        self._set_eval_times()

    def get_metrics(self):
        _, z, y_true = self.get_tracked_data()

        censor = y_true[:, 1].astype(bool)
        event_indicator = ~censor
        survival_time_true = y_true[:, 0]

        out = self.compute_surv_metrics(pred_risk_score=z,
                                        events=event_indicator,
                                        times=survival_time_true)

        out['cox_loss'] = get_cox_loss(z=z, y=y_true)

        return out


####################
# Helper functions #
####################
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def pred_clf(z, loss='cross'):
    """
    Gets classification predictions from the z input.

    Parameters
    ----------
    z: shape (n_samples, n_classes)
        The unnormalized class scores.

    Output
    ------
    y_pred, prob_pred

    y_pred: shape (n_samples, )
        The predicted class label indices.

    prob_pred: shape (n_samples, n_classes) or (n_samples, )
        The predicted probabilities for each class. Returns

    """
    # CrossEntropyLoss

    if loss == 'cross':
        prob = softmax(z, axis=1)
        y = prob.argmax(axis=1)

    else:
        prob_pos = sigmoid(z).reshape(-1, 1)
        prob = np.concatenate([1-prob_pos, prob_pos], axis=1)
        y = prob.argmax(axis=1)

    return y, prob


def pred_discr_surv(z):
    """
    Gets risk score predictions from the z input for the discrete survival loss.

    Parameters
    ----------
    z: shape (n_samples, n_bins)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).

    Output
    ------
    risk_scores: shape (n_samples, )
        The predicted risk scores.
    """
    hazards = expit(z)
    surv_func = np.cumprod(1 - hazards, axis=1)
    risk = -surv_func.sum(axis=1)
    return risk, surv_func


def get_cox_loss(z, y):
    """
    Returns the cox loss.

    Parameters
    ----------
    z: array-like, (n_samples, )

    y: array-like, (n_samples, 2)
        First column is censorship indicator.
        Second column is survival time.

    Output
    ------
    loss: float
    """

    with torch.no_grad():

        # format to torch
        z = torch.from_numpy(z)
        c_t = torch.from_numpy(y)

        # setup loss func
        loss_func = CoxLoss(reduction='sum')
        loss = loss_func(z, c_t)
        return loss.detach().cpu().numpy().item()
