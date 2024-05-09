import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

####################
# Fns for analysis #
####################
def classify_surv(risk_list, numOfclasses=2, class_labels=[]):
    """
    Given a list of predicted risks, classify them into different risk groups, based on their quantiles
    """
    if len(class_labels) == 0:
        class_labels = np.arange(numOfclasses)
    class_list, _ = pd.qcut(risk_list, q=numOfclasses, labels=class_labels, retbins=True)
    return class_list


def prepare_surv_dict(surv_list, event_list, class_list):

    classes = np.unique(class_list)
    surv_dict = {}
    event_dict = {}

    for c in classes:
        indices = np.flatnonzero(class_list == c)
        surv_dict[c] = surv_list[indices]
        event_dict[c] = event_list[indices]

    return surv_dict, event_dict


#################
# Dataframe fns #
#################
def augment_df(df, numOfaug=3):
    """
    Augment the dataframe to accommodate augmented data inputs

    Inputs
    ======
    df: Dataframe

    Returns
    =======
    df_new: New dataframe with augmented entries
    """

    pd_dict = {}
    indices = np.arange(df.shape[0])

    # Train data
    for idx in indices:
        index = df.index[idx]
        # For each augmentation
        for aug_idx in range(numOfaug + 1):
            if aug_idx == 0:
                pd_dict[index] = df.iloc[idx, :].values.tolist()
            else:
                pd_dict[index + '_aug{}'.format(aug_idx)] = df.iloc[idx, :].values.tolist()

    df_new = pd.DataFrame.from_dict(pd_dict, orient='index', columns=df.keys())

    return df_new


def load_df(csv_path, task='clf', label='BCR', days_label='BCR_days'):
    """
    Load dataframe and process the columns
    """
    df = pd.read_csv(csv_path, dtype={'patient_id': 'str'})
    df = df[df[label].notna()]  # Drop patients that do not have BCR record
    if days_label and days_label in df:
        df.loc[df[days_label].isna(), days_label] = 365 * 5 # Fill in NaNs

    print("======================")
    print("Total of {} patients".format(len(df)))

    le = LabelEncoder()
    options = df[label].unique()
    le.fit(options)
    df[label] = le.transform(df[label])

    cols2keep = ['patient_id', 'slide_id', label]
    if days_label and days_label in df:
        cols2keep.append(days_label)
    df_processed = df[cols2keep]

    df_processed = df_processed.rename(columns={label: 'event'})
    if days_label and days_label in df:
        df_processed = df_processed.rename(columns={days_label: 'event_days'})

    # Cast types
    df_processed['patient_id'] = df_processed['patient_id'].astype(str)
    df_processed['slide_id'] = df_processed['slide_id'].astype(str)
    if task=='surv':
        df_processed['event'] = df_processed['event'].astype(bool)
    else:
        df_processed['event'] = df_processed['event'].astype(int)
    df_processed = df_processed.set_index('patient_id')

    return df_processed


def stratify_df(df,
                task='clf',
                numOfbins=2,
                event_col='event',
                time_col='event_days',
                stratify_col_name='class',
                eps=0.001):
    """
    Stratify the dataframe based on non-censored survival points.
    If numOfbins=1, same stratification as the event
    """

    # Stratify uncensored patients into percentile bins
    if task=='surv':
        event_mask = df[event_col].astype(bool)
        uncensored_df = df[event_mask]
        times_no_censor = uncensored_df[time_col]

        _, q_bins = pd.qcut(times_no_censor, q=numOfbins, retbins=True, labels=False)
        q_bins[-1] = df[time_col].max() + eps
        q_bins[0] = df[time_col].min() - eps

        print(q_bins)

        # y_discrete is the index label corresponding to the discrete time interval
        y_discr, bins = pd.cut(df[time_col], bins=q_bins,
                               retbins=True, labels=False,
                               right=False, include_lowest=True)
        df['bins'] = y_discr
        df[stratify_col_name] = df['bins'].astype(str) + '_X_' + df[event_col].astype(str)

        # df[stratify_col_name] = pd.qcut(df.loc[df['event'] == 1]['event_days'],
        #                                 q=percentile_cutoff,
        #                                 labels=class_labels).astype('str')
        #
        # # Put censored patients into another class
        # df.loc[df['event'] == 0, stratify_col_name] = str(numOfbins + 1)
        # df[stratify_col_name] = df[stratify_col_name].astype('int')
        # df[stratify_col_name] -= 1

    else:  # Otherwise, the label is same as the original label class (Classification)
        df[stratify_col_name] = df[event_col]

    print(df)
    print("\n=====Class distribution=====")
    print(df[stratify_col_name].value_counts())

    return df


def load_aug_split_df(csv_path,
                      label='BCR',
                      task='clf',
                      days_label=None,
                      prop_train=0.7,
                      split_mode='loo',
                      n_splits=5,
                      numOfaug=5,
                      numOfbins=3,
                      stratify_col_name='class',
                      val_aug=False,
                      seed=10):
    """
    Load pandas dataframe and split into train/val/test (Leave One Out)

    Inputs
    ======
    prop_train: proportion of training data out of (train + val) Not the proportion out of the entire data!
    split_mode: 'loo' or 'kf'. Leave-one-out or K-fold splitting
    n_splits: number of splits, only valid for split_mode='kf'
    val_aug: (boolean) If true, also augment the validation dataset

    """

    # Load dataframe
    df_pre_aug_strat = load_df(csv_path, task, label, days_label)
    # Stratify loaded dataframe
    df_pre_aug = stratify_df(df_pre_aug_strat, task=task,
                             numOfbins=numOfbins,
                             stratify_col_name=stratify_col_name)

    # Augment the stratified dataframe
    df_aug = augment_df(df_pre_aug,
                        numOfaug=numOfaug)

    if split_mode == 'loo':
        splitter = LeaveOneOut()
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    target = df_pre_aug[stratify_col_name].values

    # Split the indices based on pre-augmented dataframe, to prevent augmented inputs of the same patients
    # ending up in different splits
    indices = np.arange(df_pre_aug.shape[0], dtype=np.int16)

    split_indices = {}
    # Split the dataset
    for split_idx, (train_val_indices, test_indices) in enumerate(splitter.split(indices, target)):
        df_train = df_pre_aug.iloc[train_val_indices]

        if prop_train == 1: # No validation dataset
            train_indices = train_val_indices
            val_indices = []
        else:   # Further split training indices into train/val
            train_indices, val_indices = train_test_split(train_val_indices,
                                                          test_size=1-prop_train,
                                                          stratify=df_train[stratify_col_name],
                                                          random_state=seed)

        # Reassign indices to accommodate for augmented dataset
        if numOfaug > 0:
            train_indices_new = []
            val_indices_new = []
            test_indices_new = []

            for idx in train_indices:
                train_indices_new.extend(np.arange((numOfaug + 1) * idx, (numOfaug + 1) * (idx + 1)))
            if len(val_indices) > 0:
                for idx in val_indices:
                    if val_aug:
                        val_indices_new.extend(np.arange((numOfaug + 1) * idx, (numOfaug + 1) * (idx + 1)))
                    else:
                        val_indices_new.extend([(numOfaug + 1) * idx])
            for idx in test_indices:
                test_indices_new.extend([(numOfaug + 1) * idx])
        else:
            train_indices_new = train_indices
            val_indices_new = val_indices
            test_indices_new = test_indices

        split_indices[split_idx] = [train_indices_new, val_indices_new, test_indices_new]

    return df_aug, split_indices

#################
# Auxiliary fns #
#################
def get_time_bins(df, numOfbins=2):
    _, bins = pd.qcut(df.loc[df['event'] == 1]['event_days'],
                      q=numOfbins,
                      retbins=True)

    return bins

def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)

def get_weights_for_balanced_clf(y):
    """
    Gets sample weights for the WeightedRandomSampler() for the data loader to make balanced training datasets in each epoch.
    Let class_counts, shape (n_classes, ) be the vector of class counts. The sample weights for the ith observation is
    n_samples / class_counts[y[i]]
    Parameters
    ----------
    y: array-like, (n_samples, )
        The observed class indices.
    Output
    ------
    sample_weights: array-like, (n_samples, )
        The sample weights
    """

    y = pd.Series(y)
    class_counts = y.value_counts()  # class counts

    n_samples = len(y)

    sample_weights = n_samples / np.array([class_counts[cl_idx]
                                           for cl_idx in y])

    return sample_weights