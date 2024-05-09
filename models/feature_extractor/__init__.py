"""
Initiates 2D/3D feature extractor, all available with pretrained weights

The current options are
- Resnet (2D/3D): The original Resnet, pretrained on Imagenet. If 3D, inflates the convolutional kernel weights across z-dimension
    - Latent dim: Depends on the Renset backbone
- SwinVIT (3D): SwinVIT, pretrained on 3D CT/MRI dataset
    - Latent dim: 768
- Res2plus1d (3D): (2+1)D CNN (with Resnet50 backbone) pretrained on Kinetics-400
    - Latent dim: 1024
    - Reference: Tran D et al., A closer look at spatiotemporal convolutions for action recognition (https://arxiv.org/pdf/1711.11248.pdf)
"""

from .ResnetInflated import resnet_3d
from .Resnet import resnet_2d
from .SwinUNETR import swin_unetr_base
from .Resnet2plus1d import resnet2plus1d
from .Swin3D import swin3d_s, swin3d_b
from .Swin2D import swin2d_s, swin2d_b


def get_extractor_model(encoder='2plus1d',
                        input=(96, 96, 96),
                        mode='3D',
                        trainable_layers=[]):
    """
    Load a feature extractor model. Can currently load resnet variants, SwinVIT, CLIP, and 2plus1d

    Args:
    - encoder (str): Name of the encoder to instatiate
    - mode (str): '2D' or '3D'
    - trainable_layers (List): List of layer names to train

    Returns:
    - model (nn.Module): Feature extractor moudule
    """

    if 'resnet' in encoder:
        # Inflated resnet
        if '3d' in encoder:
            model = resnet_3d(encoder=encoder,
                              trainable_layers=trainable_layers)
        # original resnet
        elif '2d' in encoder:
            model = resnet_2d(encoder=encoder)
        else:
            raise NotImplementedError("{} not implemented!".format(encoder))

    elif encoder == 'SwinUNETR':
        if mode == '3D':
            spatial_dims = 3
        else:
            raise NotImplementedError("Not implemented for mode {}".format(mode))

        model = swin_unetr_base(input,
                                in_channels=1,
                                trainable_layers=trainable_layers,
                                spatial_dims=spatial_dims)

    elif encoder == 'swin3d_s':
        model = swin3d_s()
    elif encoder == 'swin3d_b':
        model = swin3d_b()
    elif encoder == 'swin2d_s':
        model = swin2d_s()
    elif encoder == 'swin2d_b':
        model = swin2d_b()
    elif encoder == '2plus1d':
        assert mode == '3D', '2plus1d only works on 3D!'
        model = resnet2plus1d()
    else:
        raise NotImplementedError("{} not implemented!".format(encoder))

    return model
