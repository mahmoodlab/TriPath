"""
Main script for training the network with crossvalidation
"""

import argparse
import numpy as np
import os
import torch
import time
import yaml
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

from tqdm import tqdm
from data.ThreeDimDataset import ThreeDimFeatsBag
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from models.head import get_decoder_model, get_decoder_input_dim
from trainer.trainer_feats import train_loop, eval_loop
from utils.data_utils import load_aug_split_df, classify_surv, prepare_surv_dict
from utils.exp_utils import set_seeds, update_config, get_optim, count_parameters, EarlyStopper
from utils.eval_utils import ClfEvaler, DiscreteSurvivalEvaler, estimate_ibs
from utils.file_utils import get_exp_name
from plotter.plot_survival import plot_KM

from loss.NLLSurvLoss import NLLSurvLoss
from loss.CrossEntropyCustomLoss import CrossEntropyCustomLoss
from loss.BCELogitsCustomLoss import BCELogitsCustomLoss
from sksurv.compare import compare_survival
from sksurv.util import Surv
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


def gen_clf_plots(conf, y_true, y_pred, prob_pred, result_path):
    # Plot
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.savefig(os.path.join(result_path, 'confusion_mat.png'), bbox_inches='tight')

    if conf['numOfclasses'] == 2:  # ROC curve only makes sense for binary classification
        RocCurveDisplay.from_predictions(y_true, prob_pred[:, 1])
        plt.savefig(os.path.join(result_path, 'roc.png'), bbox_inches='tight')



# Sets the seed and creates the result folders
# @return   the device to train with, the result path, and a dictionary of the directories to use
def setup_training(conf):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(device=device, seed=conf['seed_exp'])

    exp_name = get_exp_name(conf)
    exp_path = '{}__cls-{}__split-{}-{}'.format(conf['task'],
                                                conf['numOfclasses'],
                                                conf['split_mode'],
                                                conf['split_fold'])
    result_path = os.path.join(conf['result_path'], conf['exp'], exp_path, exp_name)
    os.makedirs(result_path, exist_ok=True)

    # Save config file
    with open(os.path.join(result_path, 'conf.yaml'), 'w') as f:
        yaml.dump(conf, f)

    dirs = {'log': os.path.join(result_path, 'log'),
            'checkpoint': os.path.join(result_path, 'checkpoints')}

    for _, v in dirs.items():
        os.makedirs(v, exist_ok=True)

    return device, result_path, dirs


def get_early_stopper(stop_patience, ckpt_dir):
    """
    Set up early stopper

    Args:
    - stop_patience (int): Patience iterations
    - ckpt_dir (str): Directory to save model ckpt

    Returns:
    - early_stopper (Object): Early stopper object
    """
    # setup early stopping
    if conf['es']:
        early_stopper = EarlyStopper(save_dir=ckpt_dir,
                                     min_epoch=0,
                                     patience=stop_patience,
                                     patience_min_improve=0.001,
                                     abs_scale=True,
                                     min_good=True,  # be careful to set this!
                                     verbose=True,
                                     save_model=True
                                     )
    else:
        early_stopper = None
    return early_stopper


def get_lr_scheduler(optimizer, scheduler_name='cosine', epochs=100):
    if scheduler_name == 'constant':
        scheduler = None
    elif scheduler_name == 'cosine':
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    elif scheduler_name == 'reduce':
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5)
    else:
        raise NotImplementedError("not implemented for ", conf['scheduler'])
    return scheduler


def load_and_split(conf):
    if conf['task'] == 'clf':   # Binary classification
        clf_label = 'class'
        numOfbins = conf['numOfclasses'] - 1
    elif conf['task'] == 'surv':  # Survival (Also multiclass classification)
        clf_label = 'class'
        numOfbins = conf['numOfclasses']
    else:
        raise NotImplementedError("Not implemented for {}".format(conf['task']))

    df, split_indices = load_aug_split_df(csv_path=conf['clinical_path'],
                                          label=conf['label'],
                                          task=conf['task'],
                                          days_label=conf['days_label'],
                                          split_mode=conf['split_mode'],
                                          n_splits=conf['split_fold'],
                                          prop_train=conf['prop_train'],
                                          numOfaug=conf['numOfaug'],
                                          numOfbins=numOfbins,
                                          stratify_col_name='class',
                                          val_aug=conf['val_aug'],
                                          seed=conf['seed_data'])

    print(df)
    return df, clf_label, split_indices


def get_dataset_loaders(conf, device, df, indices, data_lists, fold_idx):
    """
    Instantiates the datasets and dataset loaders
    """
    train_indices, val_indices, test_indices = indices
    df_train = df.iloc[train_indices]
    df_val = df.iloc[val_indices]
    df_test = df.iloc[test_indices]

    data_lists['event'].extend(df_test['event'].values)
    if 'event_days' in df_test:
        data_lists['survival'].extend(df_test['event_days'].values)
    else:
        mock_surv = [1.0 if not x else np.random.random_sample() for x in df_test['event'].values]
        data_lists['survival'].extend(mock_surv)
    data_lists['indices'].extend(np.array(test_indices) / (conf['numOfaug'] + 1))
    data_lists['subject'].extend(df.iloc[test_indices].index.tolist())
    data_lists['fold'].extend([fold_idx] * len(test_indices))

    if len(df_val) == 0:  # If no validation set is assigned
        val_flag = False
    else:
        val_flag = True

    # Only subsample for the training dataset
    dataset_train = ThreeDimFeatsBag(path=conf['feats_path'],
                                     data_df=df_train,
                                     task=conf['task'],
                                     numOfaug=conf['numOfaug'],
                                     sample_prop=conf['sample_prop'],
                                     sample_mode=conf['sample_mode'],
                                     numOfclasses=conf['numOfclasses'])
    
    if val_flag:
        dataset_val = ThreeDimFeatsBag(path=conf['feats_path'],
                                       data_df=df_val,
                                       task=conf['task'],
                                       numOfclasses=conf['numOfclasses'])

    dataset_test = ThreeDimFeatsBag(path=conf['feats_path'],
                                    data_df=df_test,
                                    task=conf['task'],
                                    numOfclasses=conf['numOfclasses'])

    print("\nDATA STATISTICS")
    print("Train: {} cases (augmented {} times)".format(len(dataset_train), conf['numOfaug']))
    if val_flag:
        print("Val: {} cases (augmented {} times)".format(len(dataset_val), conf['numOfaug']))
    print("Test: {} cases ".format(len(dataset_test)))
    print(df_test)

    kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == "cuda" else {}
    loader_train = DataLoader(dataset_train, batch_size=1, sampler=None, shuffle=True, **kwargs)

    loader_val = None
    if val_flag:
        loader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                **kwargs)

    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, **kwargs)

    return loader_train, loader_val, loader_test


def train(conf,
          device,
          model,
          writer,
          loaders,
          early_stopper,
          optimizer,
          scheduler,
          loss_funcs,
          evalers,
          model_ckpt_path):
    """
    For each CV fold, train the network and evaluate on val/test dataset
    """

    loader_train = loaders['train']
    loader_val = loaders['val']
    loader_test = loaders['test']
    loss_func_train = loss_funcs['train']
    loss_func_test = loss_funcs['test']
    evaler_fold = evalers['fold']
    evaler_all = evalers['all']

    # Training loop
    train_time = []
    eval_time = []
    for epoch in tqdm(range(conf['epochs'])):
        print("============================")
        print("Epoch: {}".format(epoch))

        #######
        # Train
        print("\n*Train*")
        s = time.time()
        train_loss, train_metrics = train_loop(model=model,
                                               loader=loader_train,
                                               optimizer=optimizer,
                                               loss_func=loss_func_train,
                                               evaler=evaler_fold,
                                               grad_accum=conf['grad_accum'],
                                               scheduler=scheduler,
                                               device=device)
        e = time.time()
        train_time.append(e-s)
        if writer:
            writer.add_scalar('train/loss', train_loss, global_step=epoch)

            if train_metrics:
                if conf['task'] == 'clf':
                    writer.add_scalar('train/bal_acc', train_metrics['bal_acc'], global_step=epoch)
                    writer.add_scalar('train/auc', train_metrics['auc'], global_step=epoch)
                    writer.add_scalar('train/f1', train_metrics['f1'], global_step=epoch)

        #####
        # Val
        if loader_val is not None:
            print("\n*Validation*")
            val_loss, val_metrics, _ = eval_loop(model=model,
                                                  loader=loader_val,
                                                  loss_func=loss_func_test,
                                                  evaler=evaler_fold,
                                                  device=device)

            if writer:
                writer.add_scalar('val/loss', val_loss, global_step=epoch)

                if val_metrics:
                    if conf['task'] == 'clf':
                        writer.add_scalar('val/acc', val_metrics['acc'], global_step=epoch)
                        writer.add_scalar('val/bal_acc', val_metrics['bal_acc'], global_step=epoch)
                        writer.add_scalar('val/auc', val_metrics['auc'], global_step=epoch)
                        writer.add_scalar('val/f1', val_metrics['f1'], global_step=epoch)
                    else:
                        writer.add_scalar('val/c_index', val_metrics['c_index'], global_step=epoch)

            # check early stopping
            if conf['es']:
                stop_early = early_stopper(model=model,
                                           score=val_loss,
                                           epoch=epoch,
                                           ckpt_name=os.path.basename(model_ckpt_path))

                if stop_early:
                    print("=========================================")
                    print("Stopping early after completing {} epochs".format(epoch + 1))
                    print("=========================================")

                    # Evaluate test sample with best model
                    model.load_state_dict(torch.load(model_ckpt_path), strict=True)

                    # Fold-specific test metrics
                    eval_loop(model=model,
                              loader=loader_test,
                              loss_func=loss_func_test,
                              evaler=evaler_fold,
                              evaler_reset=True,
                              device=device,
                              verbose=True)

                    # Entire dataset test metrics
                    test_loss, test_metrics_fold, _ = eval_loop(model=model,
                                                                loader=loader_test,
                                                                loss_func=loss_func_test,
                                                                evaler=evaler_all,
                                                                evaler_reset=False,
                                                                device=device,
                                                                verbose=True)

                    break

        # Monitor loss on the test dataset
        print("\n*Test*")
        s = time.time()
        test_loss, test_metrics, _ = eval_loop(model=model,
                                                loader=loader_test,
                                                loss_func=loss_func_test,
                                                evaler=evaler_fold,
                                                evaler_reset=True,
                                                device=device,
                                                verbose=True)
        e = time.time()
        eval_time.append(e-s)

        if epoch == conf['epochs'] - 1:
            if not conf['es']:  # No early-stopping - Just save the latest model
                torch.save(model.state_dict(), model_ckpt_path)
            else:  # Early-stopping enabled, but reached maximum epoch
                model.load_state_dict(torch.load(model_ckpt_path), strict=True)

            # Fold-specific test metrics
            _, test_metrics_fold, info = eval_loop(model=model,
                                                   loader=loader_test,
                                                   loss_func=loss_func_test,
                                                   evaler=evaler_fold,
                                                   evaler_reset=True,
                                                   device=device,
                                                   verbose=True)

            # Entire dataset test metrics
            eval_loop(model=model,
                      loader=loader_test,
                      loss_func=loss_func_test,
                      evaler=evaler_all,
                      evaler_reset=False,
                      device=device,
                      verbose=True)

    return test_metrics_fold, model


def process_results(conf, results_fold, evaler_all, data_lists, result_path):
    """
    Saves/plots the experiment results
    """

    results_all = evaler_all.get_metrics()
    print(f'\neval_results aggregated: {results_all}')
    print(f'\neval_results per fold: {results_fold}')

    if conf['task'] == 'clf':
        y_true, y_pred, prob_pred = evaler_all.get_preds()
        print(f'y_pred: {y_pred}')
        print(f'y_true: {y_true}')

        # Save results
        results = {key: val for key, val in data_lists.items()}
        results['metrics_all'] = results_all
        results['metrics_fold'] = results_fold
        results['prob_pred'] = prob_pred

        with open(os.path.join(result_path, 'result.pkl'), 'wb') as f:
            pickle.dump(results, f)

        gen_clf_plots(conf, y_true, y_pred, prob_pred, result_path)

    elif conf['task'] == 'surv':
        results = {key: val for key, val in data_lists.items()}
        results['metrics_all'] = results_all
        results['metrics_fold'] = results_fold

        y_true, pred_risk, surv_func = evaler_all.get_preds()
        results['risk_pred'] = pred_risk

        with open(os.path.join(result_path, 'result.pkl'), 'wb') as f:
            pickle.dump(results, f)

    else:
        raise NotImplementedError("not implemented for ", conf['task'])


def run_training(conf, device, df, split_indices, clf_label, early_stopper, result_path, dirs):
    """
    The main training script - Initiates crossvalidation
    """
    ##########################
    # Loss function & evaler #
    ##########################
    if conf['task'] == 'clf':
        if conf['loss'] == 'cross':
            loss_func_train = CrossEntropyCustomLoss()
            loss_func_test = CrossEntropyCustomLoss()
        else:   # binary
            loss_func_train = BCELogitsCustomLoss()
            loss_func_test = BCELogitsCustomLoss()

        class_names = df[clf_label].unique()
        class_names.sort()

        evaler_fold = ClfEvaler(class_names=class_names, loss=conf['loss'])
        evaler_all = ClfEvaler(class_names=class_names, loss=conf['loss'])
        if conf['epochs_finetune'] > 0:
            evaler_all_finetune = ClfEvaler(class_names=class_names, loss=conf['loss'])

    elif conf['task'] == 'surv':
        loss_func_train = NLLSurvLoss()
        loss_func_test = NLLSurvLoss()

        evaler_fold = DiscreteSurvivalEvaler()
        evaler_all = DiscreteSurvivalEvaler()
        if conf['epochs_finetune'] > 0:
            evaler_all_finetune = DiscreteSurvivalEvaler()
    else:
        raise NotImplementedError("Loss func not implemented for {}".format(conf['task']))

    ###########
    #3 Training
    #############
    list_keys = ['event', 'survival', 'indices', 'subject', 'fold']
    data_lists = {k: [] for k in list_keys}
    results_fold = {}

    s = time.time()
    for iter_idx, (split_idx, indices) in enumerate(tqdm(split_indices.items())):
        print("**********************")
        print("Split {}".format(iter_idx))
        print("**********************")

        split_dir = os.path.join(result_path, 'split_{}').format(iter_idx)
        os.makedirs(split_dir, exist_ok=True)

        # Tensorboard writer
        writer = SummaryWriter(os.path.join(dirs['log'], "split_{}".format(iter_idx+1)), flush_secs=2)

        loader_train, loader_val, loader_test = (
            get_dataset_loaders(conf, device, df, indices, data_lists, iter_idx)
        )

        print("==================")
        print("Loading model...")
        print("==================")

        if conf['task'] == 'clf':
            if conf['loss'] == 'bce':
                out_dim = 1
            else:
                out_dim = conf['numOfclasses']
        else:
            out_dim = conf['numOfclasses']

        # Instantiate model
        conf_decoder = {
            'decoder': conf['decoder'],
            'decoder_enc': conf['decoder_enc'],
            'decoder_enc_dim': conf['decoder_enc_dim'],
            'decoder_enc_num': conf['decoder_enc_num'],
            'input_dim': get_decoder_input_dim(conf['encoder']),
            'attn_latent_dim': conf['attn_latent_dim'],
            'out_dim': out_dim,
            'dropout': conf['dropout'],
            'warm_start': conf['warm_start'],
            'context': conf['context'],
            'context_network': conf['context_network'],
        }

        set_seeds(device=device, seed=conf['seed_exp'])
        model = get_decoder_model(**conf_decoder)
        print(model)
        print("===================")
        print("Total trainable params: {}".format(count_parameters(model)))
        print("===================")
        model.to(device)

        optimizer = get_optim(model=model,
                              opt=conf['opt'],
                              lr=conf['lr'],
                              weight_decay=conf['weight_decay'])

        scheduler = get_lr_scheduler(optimizer, scheduler_name=conf['scheduler'], epochs=conf['epochs'])

        # Initialize early stopper
        if conf['es']:
            print("Resetting early stopper...")
            early_stopper.reset_tracking()

        loaders = {'train': loader_train, 'val': loader_val, 'test': loader_test}
        loss_funcs = {'train': loss_func_train, 'test': loss_func_test}
        evalers = {'fold': evaler_fold, 'all': evaler_all}
        model_ckpt_path = os.path.join(dirs['checkpoint'], 'ckpt_split--{}.pt'.format(iter_idx))
        conf_train = {'es': conf['es'], 'grad_accum': conf['grad_accum'], 'epochs': conf['epochs'], 'task': conf['task']}

        test_metrics_fold, model = train(conf_train,
                                         device,
                                         model,
                                         writer,
                                         loaders,
                                         early_stopper,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         loss_funcs=loss_funcs,
                                         evalers=evalers,
                                         model_ckpt_path=model_ckpt_path)

        #### Fine-tuning
        if conf['epochs_finetune'] > 0:
            # Unfreeze model
            flag = model.start_attention(freeze_encoder=False)
            if not flag:
                print("Activating attention not possible for the current model.. Skipping to next iteration")
                continue

            print("\n\nFinetuning the model...")
            print("Total trainable params: {}".format(count_parameters(model)))
            print("===================")

            optimizer = get_optim(model=model,
                                  opt=conf['opt'],
                                  lr=conf['lr_finetune'],
                                  weight_decay=conf['weight_decay'])

            scheduler = get_lr_scheduler(optimizer, scheduler_name=conf['scheduler'], epochs=conf['epochs_finetune'])
            conf_finetune = {'es': conf['es'],
                             'grad_accum': conf['grad_accum'],
                             'epochs': conf['epochs_finetune'],
                             'task': conf['task']}

            # Reset evalers
            evalers = {'fold': evaler_fold, 'all': evaler_all_finetune}

            # Finetune
            test_metrics_fold, model = train(conf_finetune,
                                             device,
                                             model,
                                             writer,
                                             loaders,
                                             early_stopper,
                                             optimizer=optimizer,
                                             scheduler=scheduler,
                                             loss_funcs=loss_funcs,
                                             evalers=evalers,
                                             model_ckpt_path=model_ckpt_path)

        results_fold[iter_idx] = test_metrics_fold

    e = time.time()

    for key, val in data_lists.items():
        data_lists[key] = np.array(val)

    print("====================")
    print("Finished training...")
    print("Total: {} sec".format(e-s))

    process_results(conf,
                    results_fold,
                    evaler_all_finetune if conf['epochs_finetune'] > 0 else evaler_all,
                    data_lists,
                    result_path)

def print_config(conf):
    for key, value in conf.items():
        if isinstance(value, dict):
            print(key)
            for value_key, value_value in value.items():
                print(value_key + " : " + str(value_value))
        else:
            print(key + " : " + str(value))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument('--config', type=str, default='.',
                        help='Config files that contain default parameters')
    parser.add_argument('--exp', type=str)
    parser.add_argument('--task', type=str, choices=['clf', 'surv'], default='clf',
                        help='Task for 3D')
    parser.add_argument('--mode', type=str, choices=['3D', '2D'], default='3D',
                        help='patch mode')
    parser.add_argument('--cuda', default=0, type=int,
                        choices=[0, 1, 2],
                        help='To manually parallelize the process')
    parser.add_argument('--result_path', type=str, help='Folder where the results are saved')
    parser.add_argument('--clinical_path', type=str, help='Clinical path')
    parser.add_argument('--feats_path', type=str)
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--label', type=str,
                        help='Column name from clinical path file used in classification task')
    parser.add_argument('--days_label', type=str, default='BCR_days',
                        help='Column name from clinical path file used for survival analysis, required if --task="surv"')
    parser.add_argument('--prop_train', type=float,
                        help='Train dataset proportion')
    parser.add_argument('--sample_prop', type=float)
    parser.add_argument('--sample_mode', type=str, choices=['volume', 'slice', 'seq'], default='volume',
                        help='When sampling from within each sample, whether to sample proportionally within each slice, or throughout the volume')
    parser.add_argument('--split_mode', type=str, choices=['loo', 'kf'], default='loo',
                        help='Mode for validation splits, loo = leave one out, kf = K-fold cross-validation')
    parser.add_argument('--split_fold', type=int,
                        help='For kfold, the number of folds')
    parser.add_argument('--seed_data', type=int,
                        help='The random seed for data splits')
    parser.add_argument('--seed_exp', type=int,
                        help='The random seed for rest of experiments')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs for training')
    parser.add_argument('--encoder', type=str,
                        help='Encoder used for feature extraction')
    parser.add_argument('--decoder', type=str,
                        help='Decoder for MIL')
    parser.add_argument('--opt', default='adam', type=str, help="Optimizer for training")
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--scheduler', type=str)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--grad_accum', type=int,
                        help='Gradient accumulation step')
    parser.add_argument('--numOfaug', type=int,
                        help='Number of augmentations. Assumes that the features are already augmented offline')
    parser.add_argument('--dropout', type=float,
                        help='Dropout probability')
    parser.add_argument('--decoder_enc', action='store_true', default=False)
    parser.add_argument('--decoder_enc_num', type=int)
    parser.add_argument('--decoder_enc_dim', type=int, help='Encoder dimension')
    parser.add_argument('--attn_latent_dim', type=int,
                        help='Attention latent dimension')
    parser.add_argument('--gated', action='store_true', default=False)
    parser.add_argument('--out_dim', default=2, type=int,
                        help='Output dimension')
    parser.add_argument('--context', action='store_true', default=False)
    parser.add_argument('--context_network', type=str)
    parser.add_argument('--loss', type=str,
                        help='Loss function for clf or survival task')

    parser.add_argument('--val_aug', action='store_true', default=False)

    # Early stopping
    parser.add_argument('--es', action='store_true', default=False,
                        help='Stop training early if a validation set score stops improving. If early stopping is enabled the best model (according to validation score) is checkpointed i.e. the final model is not necssarily the last epoch. If early stopping is not enabled the checkpointed model will be the model from the final epoch.')
    parser.add_argument('--stop_min', type=int)
    parser.add_argument('--stop_patience', type=int, default=15,
                        help='Number of patience steps for early stopping.')

    # Multi-class classification
    parser.add_argument('--numOfclasses', type=int, help='Number of classes for classification tasks')

    parser.add_argument('--epochs_finetune', type=int)
    parser.add_argument('--lr_finetune', type=float)
    parser.add_argument('--warm_start', action='store_true', default=False)

    args = parser.parse_args()

    conf = update_config(args) # Update args namespace with parameters in config file

    print_config(conf)

    device, result_path, dirs = setup_training(conf)
    df, clf_label, split_indices = load_and_split(conf)
    early_stopper = get_early_stopper(conf['stop_patience'], dirs['checkpoint'])
    run_training(conf, device, df, split_indices, clf_label, early_stopper, result_path, dirs)

    print_config(conf)
