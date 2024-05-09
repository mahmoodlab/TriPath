import argparse
from utils.image_gen_utils import *
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import math

def gen_risk_scores(group_counts, group_means, group_std_devs):
    risk_scores = []
    for i in range(len(group_counts)):
        mean = group_means[min(i, len(group_means)-1)]
        std_dev = group_std_devs[min(i, len(group_std_devs)-1)]
        for _ in range(group_counts[i]):
            risk_score = np.random.normal(loc=mean, scale=std_dev)
            risk_scores.append(risk_score)
    return risk_scores


def gen_survival_times(risk_scores=[1.0], time_scale=500.0, type='exp', scale=1.0, risk_coefficient=1.0):
    surv_times = []
    for risk_score in risk_scores:
        if type=='exp':
            surv_time = -1.0 * time_scale * np.log(np.random.uniform())/(scale * np.exp(risk_coefficient * risk_score))
        else:
            print('Distribution type not supported!')
        surv_times.append(surv_time)
    return surv_times


def censor_times(surv_times, ratio=0.3):
    avg_surv_time = np.mean(surv_times)
    # u_max = avg_surv_time / ratio
    qt = np.quantile(surv_times, 1.0-ratio)
    censor_time = np.random.uniform(low=min(surv_times), high=qt)
    new_surv_times = []
    count = 0.0
    for i in range(len(surv_times)):
        if censor_time > surv_times[i]:
            new_surv_times.append(surv_times[i])
        else:
            new_surv_times.append(censor_time)
            count += 1.0
    print(count/len(surv_times))
    return np.array(new_surv_times), censor_time




def create_survival_csv(class_types, class_counts, prefixes, label_counts, censored_surv_times, censor_time, surv_times, risk_scores, save_path):
    print(label_counts)
    d = {}
    names = []
    classes = []
    class_nums = []
    total = sum(class_counts)
    n_labels = label_counts[0]
    x = 0
    label = 0
    for i in range(len(class_types)):
        c_t, count, prefix = class_types[i], class_counts[i], prefixes[i]
        n_digits = len(str(count))
        for j in range(count):
            x += 1
            if x > n_labels:
                label += 1
                n_labels += label_counts[label]
            name = prefix+'_'+c_t+'_'+str(j).zfill(n_digits)
            names.append(name)
            classes.append(c_t)
            class_nums.append(label)
    d['patient_id'] = names
    d['slide_id'] = [''] * total
    d['type'] = classes
    d['label'] = class_nums
    d['BCR'] = (censored_surv_times!=censor_time).astype(int)
    d['BCR_days'] = censored_surv_times
    d['surv_times'] = surv_times
    d['risk_score'] = risk_scores
    df = pd.DataFrame(data=d)
    df.to_csv(save_path, header=True, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument('save_dir_path', type=str, default='.',
                        help='Path to the directory in which to save csv file.')
    parser.add_argument('--prefix', type=str, default='surv',
                        help='Prefix for saved csv file.')
    parser.add_argument('--group_counts', type=int, default=[25, 25, 25], nargs="+",
                        help='Number of samples per group.')
    parser.add_argument('--group_risk_means', type=float, default=[1.0, 2.2, 2.8], nargs="+",
                        help='Associated risk score mean for each class.')
    parser.add_argument('--group_risk_stddevs', type=float, default=[0.1, 0.1, 0.1], nargs="+",
                        help='Associated risk score standard deviation for each class.')
    args = parser.parse_args()


    img_dir = args.save_dir_path
    Path(img_dir).mkdir(parents=True, exist_ok=True)
    if img_dir[-1] == '/':
        filepath_prefix = img_dir + args.prefix
    else:
        filepath_prefix = img_dir + '/' + args.prefix

    risk_scores = gen_risk_scores(args.group_counts, args.group_risk_means, args.group_risk_stddevs)
    surv_times = gen_survival_times(risk_scores)
    censored_surv_times, censor_time = censor_times(surv_times, ratio=0.3)
    create_survival_csv(['cells'], [sum(args.group_counts)], [args.prefix], args.group_counts, censored_surv_times, censor_time, surv_times, risk_scores, filepath_prefix+'_clinical.csv')

    print('done')