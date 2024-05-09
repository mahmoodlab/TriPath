"""
SwinViT Transformers.
Adapted from https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain

References
==========
Tang, Y., Yang, D., Li, W., Roth, H.R., Landman, B., Xu, D., Nath, V. and Hatamizadeh, A., 2022.
Self-supervised pre-training of swin transformers for 3d medical image analysis.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 20730-20740)
"""

import torch
import torch.nn as nn
from .SwinUNETR_MONAI import SwinUNETR
from collections import OrderedDict

mini_batch_size = 5 # Split batch into smaller minibatch (prevent out-of-memory errors)

class SwinUNETR_enc(nn.Module):
    """
    SwinUNETR encoder

    Parameters
    ==========
    freeze: Boolean
        If False, partially unfreeze the encoder
    """
    def __init__(self,
                 img_size,
                 in_channels=1,
                 feature_size=48,
                 trainable_layers=[],
                 # train_ssl=False,
                 spatial_dims=3,
                 **kwargs):

        super().__init__()
        full_model = SwinUNETR(img_size=img_size,
                               in_channels=in_channels,
                               out_channels=2,
                               feature_size=feature_size,
                               drop_rate=0.0,
                               attn_drop_rate=0.0,
                               dropout_path_rate=0.0,
                               use_checkpoint=False,
                               spatial_dims=spatial_dims)

        self.swinViT = full_model.swinViT
        self.normalize = True
        # self.train_ssl = train_ssl

        self.channel = in_channels
        self.output_dim = 768

        # Freeze every parameter
        for param in self.swinViT.parameters():
            param.requires_grad = False

        if len(trainable_layers) > 0:
            self._unfreeze(trainable_layers)

    def forward(self, x):

        out = self._forward(x)
        out = nn.AdaptiveAvgPool3d(1)(out.unsqueeze(1)).squeeze()  # (768 x 3 x 3 x 3) -> (768, )

        if len(out.shape) == 1: # if batch_size 1 (squeeze removes singleton dimension)
            out = out.reshape(1, -1)

        return out

    def _forward(self, x):
        batch_size = x.shape[0]
        numOfminibatches = batch_size // mini_batch_size + 1

        # Layers 1 and 2 are memory-heavy + We don't fine-tune the parameters of the layers
        x2_agg = []
        for idx in range(numOfminibatches):
            if idx == numOfminibatches - 1:
                # Edge case where last batch is None
                if x[idx * mini_batch_size:].shape[0] == 0:
                    break
                else:
                    x0 = self.swinViT.patch_embed(x[idx * mini_batch_size:])
            else:
                x0 = self.swinViT.patch_embed(x[idx * mini_batch_size: (idx + 1) * mini_batch_size])
            x0 = self.swinViT.pos_drop(x0)
            x1 = self.swinViT.layers1[0](x0.contiguous())
            x2 = self.swinViT.layers2[0](x1.contiguous())

            x2_agg.append(x2)

            del x0, x1
            torch.cuda.empty_cache()

        x2_agg = torch.cat(x2_agg, dim=0)
        x3 = self.swinViT.layers3[0](x2_agg.contiguous())

        del x2_agg
        torch.cuda.empty_cache()

        x4 = self.swinViT.layers4[0](x3.contiguous())
        x4_out = self.swinViT.proj_out(x4, self.normalize)

        return x4_out

    # def _forward_ssl(self, x_in):
    #     features = self.swinViT(x_in, self.normalize)
    #     return features

    def load_from(self, weights):

        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )

    def load_weights(self, load_weights=True, pretrained_path=None, **kwargs):
        """
        Load pretrained weights
        """

        if load_weights:
            print("======================")
            print("Loading weights from {}".format(pretrained_path))
            od = OrderedDict()

            if 'model_swinvit.pt' in pretrained_path:
                weights = torch.load(pretrained_path)
                self.load_from(weights)

            else:
                saved_weights = torch.load(pretrained_path)

                for key, val in saved_weights.items():
                    new_key = '.'.join(key.split('.')[3:])
                    od[new_key] = val

                self.swinViT.load_state_dict(od, strict=False)

    def get_channel_dim(self):
        return self.channel

    def get_output_dim(self):
        return self.output_dim

    def _unfreeze(self, trainable_layers=[]):
        """
        Unfreeze parameters in the network
        """

        for layer in trainable_layers:
            if layer == 'all':
                for name, param in self.swinViT.named_parameters():
                    print("--- {} is now trainable".format(name))
                    param.requires_grad = True
                break

            for name, param in self.swinViT.named_parameters():
                if layer in name:
                    print("--- {} is now trainable".format(name))
                    param.requires_grad = True


def swin_unetr_base(input_size=(96, 96, 96),
                    trainable_layers=[],
                    in_channels=1,
                    # train_ssl=False,
                    spatial_dims=3):

    model = SwinUNETR_enc(img_size=input_size,
                          in_channels=in_channels,
                          feature_size=48,
                          trainable_layers=trainable_layers,
                          # train_ssl=train_ssl,
                          spatial_dims=spatial_dims)

    return model
