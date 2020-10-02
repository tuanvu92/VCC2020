# -*- coding: utf-8 -*-
""" Definition for common layers

Author: Ho Tuan Vu - Japan Advanced Institute of Science and Technology
Revision: 1.0
"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
import math


class ConvNorm(nn.Conv1d):
    """ 1D convolution layer with padding mode """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 pad_mode="same", dilation=1, groups=1, bias=True, w_init_gain='linear'):
        self.pad_mode = pad_mode
        if pad_mode is "same":
            _pad = int((dilation * (kernel_size - 1) + 1 - stride) / 2)
        elif pad_mode == "causal":
            _pad = dilation * (kernel_size - 1)
        else:
            _pad = 0
        self._pad = _pad
        super(ConvNorm, self).__init__(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=_pad,
                                       dilation=dilation,
                                       bias=bias,
                                       groups=groups)
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        """ Calculate forward propagation
        Args:
            signal (Tensor): input tensor

        Returns:
            Tensor: output tensor

        """
        conv_signal = super(ConvNorm, self).forward(signal)
        if self.pad_mode is "causal":
            conv_signal = conv_signal[:, :, :-self._pad]
        return conv_signal


class WNCell(nn.Module):
    """ WaveNet-like cell """
    def __init__(self, residual_dim, gate_dim, skip_dim=128, cond_dim=0,
                 kernel_size=3, dilation=1, pad_mode='same'):
        """ Initialize WNCell module
        Args:
            residual_dim (int): Number of channels for residual connection.
            gate_dim (int): Number of channels for gate connection.
            skip_dim (int): Number of channels for skip connection.
            cond_dim (int): Number of channels for conditioning variables.
            kernel_size (int): Size of kernel.
            dilation (int): Dilation rate.
            pad_mode (str): Padding mode:
                "same": input and output frame length is same
                "causal": output only depends on current and previous input frames

        """
        super(WNCell, self).__init__()
        self.hidden_dim = gate_dim
        self.cond_dim = cond_dim
        self.dilation = dilation

        self.in_layer = nn.Sequential(
            ConvNorm(residual_dim,
                     2 * gate_dim,
                     kernel_size=kernel_size,
                     dilation=dilation,
                     pad_mode=pad_mode),
            nn.InstanceNorm1d(2 * gate_dim, momentum=0.25),
        )
        if cond_dim > 0:
            self.conv_fuse = ConvNorm(2 * gate_dim, 2 * gate_dim, kernel_size=1,
                                      groups=2, pad_mode=pad_mode)

        self.res_layer = nn.Sequential(
            ConvNorm(gate_dim, residual_dim, kernel_size=kernel_size, pad_mode=pad_mode),
            nn.InstanceNorm1d(residual_dim, momentum=0.25)
        )

        self.skip_layer = nn.Sequential(
            ConvNorm(gate_dim, skip_dim, kernel_size=kernel_size, pad_mode=pad_mode),
            nn.InstanceNorm1d(skip_dim, momentum=0.25)
        )

    def forward(self, x, cond=None):
        """ Calculate forward propagation

        Args:
             x (Tensor): input variable
             cond (Tensor): condition variable

        Returns:
            Tensor: Output tensor for residual connection (B, residual_channels, T).
            Tensor: Output tensor for skip connection (B, skip_channels, T).

        """
        if self.cond_dim > 0:
            assert cond is not None
            acts = self.conv_fuse(self.in_layer(x) + cond)
        else:
            acts = self.in_layer(x)

        tanh_act = torch.tanh(acts[:, :self.hidden_dim, :])
        sigmoid_act = torch.sigmoid(acts[:, self.hidden_dim:, :])
        acts = tanh_act * sigmoid_act
        skip = self.skip_layer(acts)
        res = self.res_layer(acts)

        return (x + res) * math.sqrt(0.5), skip


class Jitter(nn.Module):
    """
    Jitter implementation from [Chorowski et al., 2019].
    During training, each latent vector can replace either one or both of
    its neighbors. As in dropout, this prevents the model from
    relying on consistency across groups of tokens. Additionally,
    this regularization also q latent representation stability
    over time: a latent vector extracted at time step t must strive
    to also be useful at time steps t ÃƒÂ¢Ã‹â€ Ã¢â‚¬â„¢ 1 or t + 1.
    """

    def __init__(self, probability=0.12):
        super(Jitter, self).__init__()

        self._probability = probability

    def forward(self, quantized):
        original_quantized = quantized.detach().clone()
        length = original_quantized.size(2)
        for i in range(length):
            """
            Each latent vector is replace with either of its neighbors with a certain probability
            (0.12 from the paper).
            """
            replace = [True, False][np.random.choice([1, 0], p=[self._probability, 1 - self._probability])]
            if replace:
                if i == 0:
                    neighbor_index = i + 1
                elif i == length - 1:
                    neighbor_index = i - 1
                else:
                    """
                    "We independently sample whether it is to
                    be replaced with the token right after
                    or before it."
                    """
                    neighbor_index = i + np.random.choice([-1, 1], p=[0.5, 0.5])
                quantized[:, :, i] = original_quantized[:, :, neighbor_index]

        return quantized


class VectorQuantize(nn.Module):
    """ Vector Quantization modules with straight-through trick """
    def __init__(self, emb_dim, n_emb):
        """ Initialize Vector Quantization module
        Args:
             emb_dim (int): Number of channels of embedding vector
             n_emb (int): Number of embedding in codebook

        """
        super(VectorQuantize, self).__init__()
        self.n_emb = n_emb
        self.emb_dim = emb_dim
        self.codebook = Parameter(torch.Tensor(emb_dim, n_emb).uniform_(-1/n_emb, 1/n_emb))
        self.emb_dim = emb_dim
        self.jitter = Jitter()

    def forward(self, z_e_x, jitter=False):
        """ Calculate forward propagation
        Args:
            z_e_x (Tensor): input tensor for quantization
            jitter (Bool): Set to True for using jitter

        """
        # codebook = torch.index_select(self.codebook, dim=0, index=codebook_index).squeeze(0)
        inputs_size = z_e_x.size()
        z_e_x_ = z_e_x.permute(0, 2, 1).contiguous().view(-1, self.emb_dim)
        dist2 = torch.sum(z_e_x_**2, 1, keepdim=True) \
            - 2*torch.matmul(z_e_x_, self.codebook) \
            + torch.sum(self.codebook**2, 0)
        _, z_id_flatten = torch.max(-dist2, dim=1)
        z_id = z_id_flatten.view(inputs_size[0], inputs_size[2])
        z_q_flatten = torch.index_select(self.codebook.t(), dim=0, index=z_id_flatten)
        z_q = z_q_flatten.view(inputs_size[0], inputs_size[2], self.emb_dim).permute(0, 2, 1)
        if jitter:
            z_q = self.jitter(z_q)
        return z_q, z_id


class Encoder(nn.Module):
    """ Encoder module with skip and residual connection """
    def __init__(self, input_dim, output_dim,
                 residual_dim, gate_dim, skip_dim,
                 kernel_size, down_sample_factor=2, dilation_rate=None):
        """ Initialize Encoder module

        Args:
            input_dim (int): Number of channels of input tensor
            output_dim (int): Number of channels of output tensor
            skip_dim (int): Number of channels of skip connection
            kernel_size (int): Size of kernel
            down_sample_factor: Upsample factor
            dilation_rate: List of dilation rate for WNCell

        Returns:
            Tensor: Output tensor

        """
        super().__init__()
        self.down_sample_factor = down_sample_factor
        if dilation_rate is None:
            dilation_rate = [1, 2, 4, 8, 16, 32]
        self.input_layer = nn.Sequential(
            ConvNorm(input_dim, 2*residual_dim, kernel_size=15),
            nn.GLU(dim=1)
        )
        if self.down_sample_factor > 1:
            self.down_sample = nn.ModuleList()
            assert down_sample_factor % 2 == 0
            for i in range(down_sample_factor//2):
                self.down_sample.extend([ConvNorm(residual_dim, 2*residual_dim, kernel_size=8, stride=2),
                                         nn.InstanceNorm1d(2*residual_dim, momentum=0.8),
                                         nn.GLU(dim=1)
                                         ])
            self.down_sample = nn.Sequential(*self.down_sample)

        self.WN = nn.ModuleList()
        for d in dilation_rate:
            self.WN.append(WNCell(residual_dim=residual_dim,
                                  gate_dim=gate_dim,
                                  skip_dim=skip_dim,
                                  kernel_size=kernel_size,
                                  dilation=d))

        self.output_layer = nn.Sequential(
            ConvNorm(skip_dim, 2 * output_dim, kernel_size=kernel_size),
            nn.InstanceNorm1d(2 * output_dim, momentum=0.8),
            nn.GLU(dim=1),
            ConvNorm(output_dim, output_dim, kernel_size=1)
        )

    def forward(self, x):
        """ Calculate forward propagation
        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Output tensor

        """
        h = self.input_layer(x)
        if self.down_sample_factor > 1:
            h = self.down_sample(h)
        skip = 0
        for i in range(len(self.WN)):
            h, _skip = self.WN[i](h)
            skip += _skip
        skip *= math.sqrt(1.0 / len(self.WN))
        output = self.output_layer(skip)
        return output


class Decoder(nn.Module):
    """ Decoder module with residual-skip connection """
    def __init__(self, input_dim, output_dim, cond_dim,
                 residual_dim, gate_dim, skip_dim, n_stage,
                 kernel_size, f0_cond_dim=16, n_gru=0, n_upsample_factor=2,
                 dilation_rate=None):
        """ Initialize Decoder module

        Args:
            input_dim (int): Number of channels of input tensor
            output_dim (int): Number of channels of output tensor
            cond_dim (int): Number of channels of condition tensor
            skip_dim (int): Number of channels of skip connection
            n_stage (int): Number of dilated WNCell stage
            kernel_size (int): Size of kernel
            n_gru (int): Number of bidirectional GRU layers
            n_upsample_factor: Upsample factor
            dilation_rate: List of dilation rate for WNCell

        Returns:
            Tensor: Output tensor

        """
        super(Decoder, self).__init__()
        self.residual_dim = residual_dim
        self.gate_dim = gate_dim
        if dilation_rate is None:
            dilation_rate = [1, 2, 4, 8, 16, 32]
        assert n_upsample_factor % 2 == 0
        self.input_layer = nn.Sequential(
            ConvNorm(in_channels=input_dim, out_channels=2*residual_dim, kernel_size=kernel_size),
            nn.InstanceNorm1d(2*residual_dim, momentum=0.25),
            nn.GLU(dim=1)
        )

        self.conv_fuse = ConvNorm(residual_dim, residual_dim, kernel_size=kernel_size)
        self.upsample = nn.ModuleList()
        for i in range(n_upsample_factor // 2):
            self.upsample.append(nn.Sequential(nn.ConvTranspose1d(residual_dim,
                                                                  2*residual_dim,
                                                                  kernel_size=8,
                                                                  stride=2,
                                                                  padding=3),
                                               nn.InstanceNorm1d(2*residual_dim, momentum=0.25),
                                               nn.GLU(dim=1)))

        self.WN = nn.ModuleList()
        for i in range(n_stage):
            for d in dilation_rate:
                self.WN.append(WNCell(residual_dim=residual_dim,
                                      gate_dim=gate_dim,
                                      skip_dim=skip_dim,
                                      cond_dim=cond_dim,
                                      kernel_size=kernel_size,
                                      dilation=d))
        self.cond_layer = nn.Sequential(
            nn.Linear(cond_dim, residual_dim + 2 * gate_dim * len(self.WN)),
            nn.ReLU()
        )

        self.f0_layer = ConvNorm(f0_cond_dim, residual_dim + 2 * gate_dim * len(self.WN))

        self.gru = None
        if n_gru > 0:
            self.gru = nn.GRU(skip_dim, skip_dim//2, n_gru,
                              batch_first=True, bidirectional=True)

        self.output_layer = nn.Sequential(
            ConvNorm(skip_dim, 2 * output_dim, kernel_size=kernel_size),
            nn.InstanceNorm1d(2 * output_dim, momentum=0.25),
            nn.GLU(dim=1),
            ConvNorm(output_dim, output_dim, kernel_size=15),
        )

    def forward(self, x, cond_in, f0):
        """ Calculate forward pass

        Args:
            x (Tensor): input tensor
            cond_in (Tensor): condition tensor

        Returns:
            Tensor: Output tensor
        """
        cond = self.cond_layer(cond_in).unsqueeze(-1) + self.f0_layer(f0)
        # h = self.conv_fuse(self.input_layer(x) + cond[:, :self.residual_dim])
        h = self.input_layer(x)
        for upsample in self.upsample:
            h = upsample(h)
        skip = 0
        for i in range(len(self.WN)):
            h, _skip = self.WN[i](h, cond[:, self.residual_dim + 2*i*self.gate_dim:
                                          self.residual_dim + 2*(i+1)*self.gate_dim])
            skip += _skip
        # Normalizing value
        skip *= math.sqrt(1.0 / len(self.WN))
        if self.gru is not None:
            skip, _ = self.gru(skip.transpose(1, 2))
            skip = skip.transpose(1, 2)
        output = self.output_layer(skip)

        return output
