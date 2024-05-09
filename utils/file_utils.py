import h5py
from datetime import datetime


def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a'):
    """
    Save the dictionary to h5 file
    """
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path


def get_exp_name(conf):
    """
    Create folder name for the experiment

    Args:
    - conf (dict): Dictionary with experiment-related parameters

    Returns:
    - exp (str): Folder name
    """
    now = datetime.now()

    if "MIL" in conf['exp']:
        name_dict = {
            'exp': conf['exp'],
            'patch_config': conf['feats_path'].split('/')[-3],
            'pretrain': conf['feats_path'].split('/')[-1],
            'seed': conf['seed_data'],
            'encoder': conf['encoder'],
            'decoder': conf['decoder'],
            'aug': conf['numOfaug'],
            'sample_prop': conf['sample_prop'],
            'accum': conf['grad_accum'],
            'enc_dim': conf['decoder_enc_dim'],
            'attn_latent_dim': conf['attn_latent_dim'],
            'dropout': conf['dropout'],
            'decay': conf['weight_decay'],
            'time': now.strftime("%Y%m%d-%H%M%S"),
            'epochs_ft': conf['epochs_finetune'],
            'lr_ft': conf['lr_finetune']
        }

        exp = 'seed--{seed}__{patch_config}__decay--{decay}'\
                '__drop--{dropout}__prop--{sample_prop}__enc--{encoder}--{pretrain}__dec--{decoder}--{enc_dim}--{attn_latent_dim}'\
                '__accum--{accum}_aug--{aug}__{time}__ft--{epochs_ft}'.format(**name_dict)

    elif conf['exp'] == 'baseline':
        enc = '_'.join(conf['feats_path'].split('/')[-2].split('_')[:-3])
        pretrain = conf['feats_path'].split('/')[-1]

        exp = '{}__model--{}__aug--{}__enc--{}--{}'\
            '__seed--{}__summary--{}'.format(conf['exp'], conf['baseline_model'],
                                                       conf['numOfaug'],
                                                       enc,
                                                       pretrain,
                                                       conf['seed'],
                                                       conf['summary_stat'])

    else:
        raise NotImplementedError("Not implemented!")

    return exp