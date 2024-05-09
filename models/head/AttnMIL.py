
import torch
import torch.nn as nn
import torch.nn.functional as F
from numbers import Number

class AttnMeanPoolMIL(nn.Module):
    """
    Attention mean pooling architecture of (Ilse et al, 2018).

    """

    def __init__(self,
                 gated=True,
                 encoder_dim=256,
                 attn_latent_dim=128,
                 dropout=0.5,
                 out_dim=1,
                 activation='softmax',
                 encoder=None,
                 warm_start=False):
        super().__init__()

        # setup attention mechanism
        if gated:
            attention = GatedAttn(n_in=encoder_dim,
                                  n_latent=attn_latent_dim,
                                  dropout=dropout)

        else:
            attention = Attn(n_in=encoder_dim,
                             n_latent=attn_latent_dim,
                             dropout=dropout)

        self.warm_start = warm_start
        self.attention = attention

        self.activation = activation

        if self.warm_start:
            for param in self.attention.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(nn.Linear(encoder_dim, out_dim))

        if encoder is None:
            self.encoder = nn.Identity()
        else:
            self.encoder = encoder

    def start_attention(self, freeze_encoder=True, **kwargs):
        """
        Turn on attention & freeze encoder if necessary
        """
        for param in self.attention.parameters():
            param.requires_grad = True

        self.warm_start = False

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        return True

    def attend(self, x, coords=None):
        """
        Given input, compute attention scores

        Returns:
        - attn_scores: tensor (n_batches, n_instances, n_feats) Unnormalized attention scores
        """

        n_batches, n_instances, _ = x.shape
        if self.warm_start:
            attn_scores = torch.ones((n_batches, n_instances, 1), device=x.device)
        else:
            attn_scores = self.attention(x) # (n_batches * n_instances, 1)

        attn_scores = attn_scores.view(n_batches, n_instances, 1) # unflatten

        if self.activation == 'softmax':
            attn = F.softmax(attn_scores, dim=1) #(n_batches, n_instances, 1)
        else:
            attn = F.relu(attn_scores)

        bag_feats = x.view(n_batches, n_instances, -1)  # unflatten

        out = (bag_feats * attn).sum(1)

        return attn_scores, out

    def encode(self, x):
        x_enc = self.encoder(x)
        return x_enc

    def forward(self, x, coords):
        """
        Args:
        - x (n_batches x n_instances x feature_dim): input instance features
        - coords (list of tuples): coordinates. Not really used for this vanilla version

        Returns:
        - out (n_batches x encode_dim): Aggregated features
        - attn_dict (dict): Dictionary of attention scores
        """
        n_batches, n_instances, _ = x.shape
        attn_dict = {'inter': [], 'intra': []}

        x_enc = self.encoder(x)
        attn, out = self.attend(x_enc)  # (n_batches * n_instances, 1), (n_batches, encode_dim)

        out = self.head(out)

        attn_dict['intra'] = attn.detach()
        levels = torch.unique(coords[:, 0])
        attn_dict['inter'] = torch.ones(n_batches, len(levels), 1)    # All slices receive same attn

        return out, attn_dict

    def captum(self, x):
        """
        For computing IG scores with captum. Very similar to forward function
        """
        n_batches, n_instances, _ = x.shape

        x_enc = self.encoder(x)
        attn, out = self.attend(x_enc)  # (n_batches * n_instances, 1), (n_batches, encode_dim)

        out = self.head(out)

        return out


class SumMIL(nn.Module):
    """
    Similar to AttnMeanPoolMIL, but with uniform attention scores for all instances
    """
    def __init__(self,
                 encoder_dim=256,
                 out_dim=1,
                 encoder=None):
        super().__init__()

        self.head = nn.Sequential(nn.Linear(encoder_dim, out_dim))

        if encoder is None:
            self.encoder = nn.Identity()
        else:
            self.encoder = encoder

    def start_attention(self, freeze_encoder=True, **kwargs):
        """
        Turn on attention & freeze encoder if necessary
        """
        return False

    def attend(self, x, coords):
        """
        Given input, compute attention scores

        Returns
        =======
        attn_scores: tensor (n_batches, n_instances, n_feats)
            Uniform attention scores
        """
        n_batches, n_instances, _ = x.shape
        attn_scores = torch.ones_like(x)  # (n_batches, n_instances, n_feats)
        attn = F.softmax(attn_scores, dim=1)

        bag_feats = x.view(n_batches, n_instances, -1)  # unflatten

        out = (bag_feats * attn).sum(1)

        return attn_scores, out

    def encode(self, x):
        x_enc = self.encoder(x)
        return x_enc

    def forward(self, x, coords):
        n_batches, n_instances, _ = x.shape
        attn_dict = {'inter': [], 'intra': []}

        x_enc = self.encoder(x)
        attn, out = self.attend(x_enc, coords)  # (n_batches * n_instances, 1), (n_batches, encode_dim)
        out = self.head(out)

        attn_dict['intra'] = attn.detach()
        levels = torch.unique(coords[:, 0])
        attn_dict['inter'] = torch.ones(n_batches, len(levels), 1)  # All slices receive same attn

        return out, attn_dict

    def captum(self, x):
        """
        For computing IG scores with captum
        """
        n_batches, n_instances, _ = x.shape

        x_enc = self.encoder(x)
        attn, out = self.attend(x_enc)  # (n_batches * n_instances, 1), (n_batches, encode_dim)

        out = self.head(out)

        return out


class HierarchicalAttnMeanPoolMIL(nn.Module):
    """
    Hierarchical Attention mean pooling architecture. There are several possible modes, which can all be explored

    attn_inter_mode: str
        'avg': Take average of all the attended slices for block-level representation
        'max': Take maximum of all the attended slices for block-level representation
        'attn': Use (non-gated) attention mechanism
        'gated_attn': Use gate attention mechanism
    """

    def __init__(self,
                 gated=True,
                 encoder_dim=256,
                 attn_latent_dim=128,
                 attn_inter_mode='avg',
                 dropout=0.5,
                 out_dim=1,
                 encoder=None,
                 context=False,
                 context_network='GRU',
                 warm_start=False):
        super().__init__()

        # setup intra (within slice) attention mechanism
        if gated:
            attention_intra = GatedAttn(n_in=encoder_dim,
                                        n_latent=attn_latent_dim,
                                        dropout=dropout)

        else:
            attention_intra = Attn(n_in=encoder_dim,
                                   n_latent=attn_latent_dim,
                                   dropout=dropout)

        self.attn_inter_mode = attn_inter_mode
        if attn_inter_mode == 'attn':
            attention_inter = Attn(n_in=encoder_dim,
                                   n_latent=attn_latent_dim,
                                   dropout=dropout)
        elif attn_inter_mode == 'gated':
            attention_inter = GatedAttn(n_in=encoder_dim,
                                        n_latent=attn_latent_dim,
                                        dropout=dropout)
        else:
            attention_inter = None

        self.warm_start_intra = warm_start
        self.warm_start_inter = warm_start

        self.attention_intra = attention_intra
        self.attention_inter = attention_inter

        if self.warm_start_intra:
            if self.attention_intra is not None:
                for param in self.attention_intra.parameters():
                    param.requires_grad = False

        if self.warm_start_inter:
            if self.attention_inter is not None:
                for param in self.attention_inter.parameters():
                    param.requires_grad = False

        self.head = nn.Sequential(nn.Linear(encoder_dim, out_dim))

        # concat encoder and attention
        if encoder is None:
            self.encoder = nn.Identity()
        else:
            self.encoder = encoder

        self.context = context
        if context:
            if context_network == 'GRU':
                self.build_context = nn.GRU(input_size=encoder_dim,
                                            hidden_size=int(0.5 * encoder_dim),
                                            dropout=dropout,
                                            bidirectional=True)
            elif context_network == 'RNN':
                self.build_context = nn.RNN(input_size=encoder_dim,
                                            hidden_size=int(0.5 * encoder_dim),
                                            dropout=dropout,
                                            bidirectional=True)
            else:
                raise NotImplementedError('Context network {} not implemented'.format(context_network))
        else:
            self.build_context = nn.Identity()

    def start_attention(self, freeze_encoder=True, attn_dict={'intra': True, 'inter': True}):
        """
        Turn on attention & freeze encoder if necessary
        """

        if attn_dict['intra'] is True:
            print("\nActiviating intra attention...")
            if self.attention_intra is not None:
                for param in self.attention_intra.parameters():
                    param.requires_grad = True
            self.warm_start_intra = False  # Turn off the warm start flag

        if attn_dict['inter'] is True:
            print("\nActiviating inter attention...")
            if self.attention_inter is not None:
                for param in self.attention_inter.parameters():
                    param.requires_grad = True
            self.warm_start_inter = False

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        return True

    def attend_intra(self, x, coords=None):
        """
        Attend patches within each slice
        """
        # Assume single batch for now
        n_batches = x.shape[0]
        levels = torch.unique(coords[:, 0])

        #########################
        # Intra-slice attention #
        #########################
        attn_intra = []
        latent_intra = []

        for lev in levels:
            indices = torch.nonzero(coords[:, 0] == lev).flatten()
            n_instances = len(indices)

            # (n_batches * n_instances, 1), (n_batches * n_instances, encoder_dim)
            if self.warm_start_intra:
                attn_scores = torch.ones((n_batches, n_instances, 1), device=x.device)
                attn_scores = attn_scores.view(n_batches, n_instances, 1)  # unflatten
            else:
                attn_scores = self.attention_intra(x[:, indices])

            # unflatten
            bag_feats = x[:, indices].view(n_batches, n_instances, -1)
            attn_scores = attn_scores.view(n_batches, n_instances, 1)

            attn = F.softmax(attn_scores, dim=1)
            out = (bag_feats * attn).sum(1)

            latent_intra.append(out)
            attn_intra.append(attn_scores)

        out = torch.concat(latent_intra, dim=0)
        out = out.view(n_batches, len(levels), -1)  # (n_batches, n_levels, encoder_dim)

        attn_intra = torch.concat(attn_intra, dim=1)    # (n_batches, n_levels, encoder_dim)

        return attn_intra, out

    def attend_inter(self, x, coords=None):
        """
        Attend across slices
        """
        #################
        # Build context #
        #################

        latent_intra = self.build_context(x)[0] if self.context else x

        #########################
        # Inter-slice attention #
        #########################
        n_batches, n_levels, _ = latent_intra.shape

        if self.warm_start_inter:
            attn_scores = torch.ones((n_batches, n_levels, 1), device=x.device)
            attn_scores = attn_scores.view(n_batches, n_levels, 1)  # unflatten
            attn = F.softmax(attn_scores, dim=1)  # (n_batches, n_levels, 1)
            out = (latent_intra * attn).sum(1)
        else:
            if self.attn_inter_mode == 'avg':
                attn_scores = torch.ones((n_batches, n_levels, 1), device=x.device)
                attn_scores = attn_scores.view(n_batches, n_levels, 1)  # unflatten

                out = nn.AdaptiveAvgPool2d((1, None))(latent_intra)
                out = out.squeeze(dim=1)
            elif self.attn_inter_mode == 'max':
                attn_scores = torch.ones((n_batches, n_levels, 1), device=x.device)
                attn_scores = attn_scores.view(n_batches, n_levels, 1)  # unflatten

                out = nn.AdaptiveMaxPool2d((1, None))(latent_intra)
                out = out.squeeze(dim=1)
            else:   # Adaptive attention
                # (n_batches * n_levels, 1), (n_batches * n_levels, encode_dim)
                attn_scores = self.attention_inter(latent_intra)

                # unflatten
                attn_scores = attn_scores.view(n_batches, n_levels, 1)
                attn = F.softmax(attn_scores, dim=1)  # (n_batches, n_levels, 1)
                out = (latent_intra * attn).sum(1)

        return attn_scores, out

    def attend(self, x, coords, mode='inter'):
        """
        Attend 1) patches within each slice (intra-slice) or 2) slices (inter-slice)
        """
        if mode == 'intra':
            attn, out = self.attend_intra(x, coords)
        elif mode == 'inter':
            attn, out = self.attend_inter(x, coords)
        else:
            raise NotImplementedError("Not implemented")

        return attn, out

    def encode(self, x):
        x_enc = self.encoder(x)
        return x_enc

    def forward(self, x, coords):
        coords = coords.squeeze()

        attn_dict = {'inter': [], 'intra': []}

        x_enc = self.encoder(x)
        attn_intra, out_intra = self.attend(x_enc, coords, mode='intra')  # (n_batches * n_instances, 1), (n_batches, encode_dim)
        attn_inter, out = self.attend(out_intra, coords, mode='inter')
        out = self.head(out)

        attn_dict['intra'] = attn_intra.detach()
        attn_dict['inter'] = attn_inter.detach() if attn_inter is not None else None

        return out, attn_dict

    def captum(self, x, coords=None):
        """
        Still under development
        """
        coords = coords.squeeze()

        x_enc = self.encoder(x)
        attn_intra, out_intra = self.attend(x_enc, coords,
                                            mode='intra')  # (n_batches * n_instances, 1), (n_batches, encode_dim)
        attn_inter, out = self.attend(out_intra, coords, mode='inter')
        out = self.head(out)

        return out


class Attn(nn.Module):
    """
    The attention mechanism from Equation (8) of (Ilse et al, 2008).

    Args:
    - n_in (int): Number of input dimensions.
    - n_latent (int or None): Number of latent dimensions. If None, will default to (n_in + 1) // 2.
    - dropout: (bool, float): Whether or not to use dropout. If True, will default to p=0.25

    References
    ----------
    Ilse, M., Tomczak, J. and Welling, M., 2018, July. Attention-based deep multiple instance learning. In International conference on machine learning (pp. 2127-2136). PMLR.
    """

    def __init__(self, n_in, n_latent=None, dropout=False):
        super().__init__()

        if n_latent is None:
            n_latent = (n_in + 1) // 2

        # basic attention scoring module
        self.score = [nn.Linear(n_in, n_latent),
                      nn.Tanh(),
                      nn.Linear(n_latent, 1)]

        # maybe add dropout
        if dropout:
            if isinstance(dropout, Number):
                p = dropout
            else:
                p = 0.25

            self.score.append(nn.Dropout(p))

        self.score = nn.Sequential(*self.score)

    def forward(self, x):
        """
        Outputs normalized attention.

        Args:
        - x (n_batches, n_instances, n_in) or (n_instances, n_in): The bag features.

        Returns:
        - attn_scores (n_batches, n_instances, 1) or (n_insatnces, 1):
            The unnormalized attention scores.

        """
        attn_scores = self.score(x)

        return attn_scores


class GatedAttn(nn.Module):
    """
    The gated attention mechanism from Equation (9) of (Ilse et al, 2008).
    Parameters
    ----------
    n_in: int
        Number of input dimensions.
    n_latent: int, None
        Number of latent dimensions. If None, will default to (n_in + 1) // 2.
    dropout: bool, float
        Whether or not to use dropout. If True, will default to p=0.25
    References
    ----------
    Ilse, M., Tomczak, J. and Welling, M., 2018, July. Attention-based deep multiple instance learning. In International conference on machine learning (pp. 2127-2136). PMLR.
    """

    def __init__(self, n_in, n_latent=None, dropout=False):
        super().__init__()

        if n_latent is None:
            n_latent = (n_in + 1) // 2

        self.tanh_layer = [nn.Linear(n_in, n_latent),
                           nn.Tanh()]

        self.sigmoid_layer = [nn.Linear(n_in, n_latent),
                              nn.Sigmoid()]

        if dropout:
            if isinstance(dropout, Number):
                p = dropout
            else:
                p = 0.25

            self.tanh_layer.append(nn.Dropout(p))
            self.sigmoid_layer.append(nn.Dropout(p))

        self.tanh_layer = nn.Sequential(*self.tanh_layer)
        self.sigmoid_layer = nn.Sequential(*self.sigmoid_layer)

        self.w = nn.Linear(n_latent, 1)

    def forward(self, x):
        """
        Outputs normalized attention.

        Args:
        - x (n_batches, n_instances, n_in) or (n_instances, n_in): The bag features.

        Returns:
        - attn_scores (n_batches, n_instances, 1) or (n_insatnces, 1):
            The unnormalized attention scores.
        """

        attn_scores = self.w(self.tanh_layer(x) * self.sigmoid_layer(x))

        return attn_scores