# -*- coding: utf-8 -*-
""" Hierarchical Vector-Quantized Variational Autoencoder model

Author: Ho Tuan Vu - Japan Advanced Institute of Science and Technology
Revision: 1.0

"""

import numpy as np
import torch
import torch.nn as nn
from modules.layers import Encoder, Decoder, VectorQuantize, ConvNorm


class VQVAE2Stage(nn.Module):
    """ Vector-Quantized VAE with 3-stage hierarchical structure """
    def __init__(self, mel_dim=80, speaker_emb_dim=16, n_speaker=100,
                 use_c0=True, jitter=False, norm=False, mcc_stats_file=None,
                 quantize_configs=None, encoder_configs=None, decoder_configs=None):
        """ Initialized VQ-VAE modules
        Args:
            mel_dim (int): Number of channels of mel-spectrogram
            speaker_emb_dim (int): Number of channels of speaker embeddings
            n_speaker (int): Number of speakers
            use_c0 (bool):
                True: Output the first cepstrum (c0)
                False: Copy the first cepstrum (c0) to output
            jitter (bool): Use jitter in quantization layers
            quantize_configs: Configuration for quantization layers
            encoder_configs: Configuration for encoder modules
            decoder_configs: Configuration for decoder modules

        """
        super().__init__()
        assert encoder_configs is not None
        assert decoder_configs is not None
        assert quantize_configs is not None
        self.n_speaker = n_speaker
        self.speaker_emb_dim = speaker_emb_dim
        self.use_c0 = use_c0
        self.mel_dim = mel_dim
        self.jitter = jitter
        self.norm = norm
        if self.norm:
            assert mcc_stats_file is not None
            mcc_stats = np.load(mcc_stats_file)
            self.mcc_mean = torch.from_numpy(mcc_stats[0]).float().unsqueeze(0).unsqueeze(-1).cuda()
            self.mcc_scale = torch.from_numpy((mcc_stats[1])).float().unsqueeze(0).unsqueeze(-1).cuda()

        self.encoder_bot = Encoder(input_dim=mel_dim, **encoder_configs["bot"])
        self.encoder_top = Encoder(input_dim=encoder_configs["bot"]["output_dim"],
                                   output_dim=quantize_configs["top"]["emb_dim"],
                                   **encoder_configs["top"])
        self.f0_encoder_bot = ConvNorm(1, 16, kernel_size=5, stride=1)
        self.f0_encoder_top = ConvNorm(16, 16, kernel_size=8, stride=2)

        self.decoder_top = Decoder(input_dim=quantize_configs["top"]["emb_dim"],
                                   **decoder_configs["top"])
        self.decoder_bot = Decoder(input_dim=quantize_configs["bot"]["emb_dim"] + decoder_configs["top"]["output_dim"],
                                   output_dim=mel_dim,
                                   **decoder_configs["bot"])

        self.quantize_conv_bot = ConvNorm(in_channels=encoder_configs["bot"]["output_dim"],
                                          out_channels=quantize_configs["bot"]["emb_dim"],
                                          kernel_size=3)

        self.quantize_top = VectorQuantize(**quantize_configs["top"])
        self.quantize_bot = VectorQuantize(**quantize_configs["bot"])
        self.speaker_emb_layer = nn.Linear(n_speaker, speaker_emb_dim, bias=False)

    def forward(self, inputs):
        """ Calculate loss for training VQ-VAE model
        Args:
            inputs (list of Tensor): Batch data contains mel-cepstrum and target id

        Returns:
            Tensor: sum of all loss components
            List of Tensor: loss components

        """
        x, speaker_id = inputs
        mcc = x[:, :-1]
        f0 = x[:, -1:]
        f0_bot = self.f0_encoder_bot(f0)
        f0_top = self.f0_encoder_top(f0_bot)

        mcc_norm = self.normalize(mcc)
        s = self.speaker_emb_layer(speaker_id)
        z, z_id, encoder_loss, perplexity = self.encode(mcc_norm, return_loss=True)
        mcc_hat_norm = self.decode(z, s, f0_top, f0_bot)
        mcc_hat = self.denormalize(mcc_hat_norm)

        if self.use_c0:
            mel_hat = self.mcc2mel(mcc_hat)
        else:
            # Copy the first channel from input
            mel_hat = self.mcc2mel(torch.cat([mcc[:, :1], mcc_hat[:, 1:]], dim=1))

        mel_src = self.mcc2mel(mcc)
        if self.use_c0:
            rc_loss = (mcc - mcc_hat).pow(2).mean()
        else:
            # Ignore the first channel
            rc_loss = (mcc_norm[:, 1:] - mcc_hat_norm[:, 1:]).pow(2).mean()
        mel_loss = (mel_src - mel_hat).pow(2).mean()
        vq_loss, commitment_loss = encoder_loss
        loss = rc_loss + vq_loss + 0.25 * commitment_loss + 0.1 * mel_loss
        return loss, [rc_loss, mel_loss, vq_loss, commitment_loss, perplexity]

    def inference(self, inputs, return_zid=False, input_speaker_emb=False):
        """ Calculate inference
        Args:
            inputs (list of Tensor): contains mel-cepstrum and target speaker id
            return_zid (bool): include z_id in return
        Returns:
            Tensor (B, mel_dim, Frames): inferred mel-spectrum

        """
        if input_speaker_emb:
            x, s = inputs
        else:
            x, speaker_id = inputs
            s = self.speaker_emb_layer(speaker_id)
        mcc = x[:, :-1]
        f0 = x[:, -1:]
        f0_bot = self.f0_encoder_bot(f0)
        f0_top = self.f0_encoder_top(f0_bot)

        mcc_norm = self.normalize(mcc)
        z, z_id = self.encode(mcc_norm, training=False)
        mcc_hat_norm = self.decode(z, s, f0_top, f0_bot)
        mcc_hat = self.denormalize(mcc_hat_norm)

        if self.use_c0:
            mel_hat = self.mcc2mel(mcc_hat)
        else:
            # Copy the first channel from input
            mel_hat = self.mcc2mel(torch.cat([mcc[:, :1], mcc_hat[:, 1:]], dim=1))
        if return_zid:
            return mel_hat, z_id
        else:
            return mel_hat

    def encode(self, x, return_loss=False, training=True):
        """ Calculate encode pass
        Args:
            x (Tensor): input tensor
            return_loss (bool): return the loss for training phase
            training (bool): flag indicate training phase

        Returns:
            Lists of Tensor: Latent variables and loss components (if return_loss is True)

        """
        h_bot = self.encoder_bot(x)
        h_top = self.encoder_top(h_bot)

        z_top, z_id_top = self.quantize_top(h_top, (training and self.jitter))
        z_top_st = h_top + (z_top - h_top).detach()

        h_bot_q = self.quantize_conv_bot(h_bot)
        z_bot, z_id_bot = self.quantize_bot(h_bot_q, (training and self.jitter))
        z_bot_st = h_bot_q + (z_bot - h_bot_q).detach()

        if return_loss:
            commitment_loss = (h_top - z_top.detach()).pow(2).mean() + \
                              (h_bot_q - z_bot.detach()).pow(2).mean()

            vq_loss = (h_top.detach() - z_top).pow(2).mean() + \
                      (h_bot_q.detach() - z_bot).pow(2).mean()
            perplexity_top = self.calculate_perplexity(z_id_top, self.quantize_top.n_emb)
            perplexity_bot = self.calculate_perplexity(z_id_bot, self.quantize_bot.n_emb)

            return [z_top_st, z_bot_st], \
                   [z_id_top.detach(), z_id_bot.detach()], \
                   [vq_loss, commitment_loss], \
                   [perplexity_top, perplexity_bot]
        else:
            return [z_top_st, z_bot_st], [z_id_top, z_id_bot]

    def decode(self, z, s, f0_top, f0_bot):
        """ Calculate decode pass
        Args:
            z (Tensor): latent variable
            s (Tensor): speaker embedding
            f0_top (Tensor): F0 top
            f0_bot (Tensor): F0 bot

        Returns:
            Tensor: output tensor

        """
        z_top, z_bot = z
        z_top_dec = self.decoder_top(z_top, s, f0_top)
        x_hat = self.decoder_bot(torch.cat([z_bot,
                                            z_top_dec],
                                           dim=1),
                                 s, f0_bot)
        return x_hat

    def normalize(self, mcc):
        if self.norm:
            mcc = (mcc - self.mcc_mean) / self.mcc_scale
        return mcc

    def denormalize(self, mcc):
        if self.norm:
            mcc = (mcc * self.mcc_scale) + self.mcc_mean
        return mcc

    def get_speaker_emb(self):
        """ Get speaker embedding """
        return self.speaker_emb_layer.weight.data.T

    def set_speaker_emb(self, speaker_emb):
        """ Set speaker embedding
        Args:
            speaker_emb (Tensor): input speaker embeddings
        """
        self.speaker_emb_layer.weight = nn.Parameter(speaker_emb)

    def get_codebook(self):
        codebook_top = self.quantize_top.codebook.data
        codebook_bot = self.quantize_bot.codebook.data
        return codebook_top, codebook_bot

    def set_codebook(self, codebook_top, codebook_bot):
        assert self.quantize_top.codebook.shape == codebook_top.shape
        assert self.quantize_bot.codebook.shape == codebook_bot.shape
        self.quantize_top.codebook = nn.Parameter(codebook_top)
        self.quantize_bot.codebook = nn.Parameter(codebook_bot)

    @staticmethod
    def calculate_perplexity(z_id, codebook_size):
        z_id = z_id.flatten()
        z_id_onehot = torch.eye(codebook_size, dtype=torch.float32).cuda().index_select(dim=0, index=z_id)
        avg_probs = z_id_onehot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity

    @staticmethod
    def mel2mcc(mel):
        mel_ndim = mel.shape[1]
        mel = mel.transpose(1, 2).unsqueeze(-1)
        mel = torch.cat([mel, torch.zeros_like(mel)], dim=-1)
        mcc = torch.irfft(mel, signal_ndim=1, signal_sizes=(2 * (mel_ndim - 1),)).transpose(1, 2)[:, :mel_ndim]
        mcc[:, 0] /= 2.
        return mcc

    @staticmethod
    def mcc2mel(mcc):
        mcc = mcc.transpose(1, 2)
        mcc = torch.cat([mcc, torch.flip(mcc[:, :, 1:-1], dims=[-1])], dim=-1)
        mcc[:, :, 0] = mcc[:, :, 0] * 2.0
        mel = torch.rfft(mcc, signal_ndim=1)
        mel = mel[:, :, :, 0]
        return mel.transpose(1, 2)
