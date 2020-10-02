# -*- coding: utf-8 -*-
""" Speech synthesizer modules

Author: Ho Tuan Vu - Japan Advanced Institute of Science and Technology
Revision: 1.0

"""

import torch
from parallel_wavegan.models.parallel_wavegan import ParallelWaveGANGenerator
from sklearn.preprocessing import StandardScaler
from utils.common_utils import read_hdf5


class PWGSynthesizer(object):
    """ Synthesizer with ParallelWaveGAN
    This code is implemented for used with this repo: https://github.com/kan-bayashi/ParallelWaveGAN
    """
    def __init__(self, checkpoint_path, stats_file, generator_params):
        """ Initialize PWG

        Args:
            checkpoint_path: Parallel WaveGAN checkpoint file
            stats_file: statistic file (h5)

        """
        wavegan_state_dict = torch.load(checkpoint_path)
        self.wavegan = ParallelWaveGANGenerator(**generator_params)
        self.wavegan.load_state_dict(wavegan_state_dict["model"]["generator"])
        self.wavegan.remove_weight_norm()
        self.wavegan.eval()
        self.wavegan_scaler = StandardScaler()
        self.wavegan_scaler.mean_ = read_hdf5(stats_file, "mean")
        self.wavegan_scaler.scale_ = read_hdf5(stats_file, "scale")
        self.wavegan_scaler.n_features_in_ = self.wavegan_scaler.mean_.shape[0]

    def synthesize(self, mel):
        """ Synthesize waveform

        Args:
            mel (ndarray): log Mel-spectrogram

        Returns:
            ndarray: waveform

        """
        assert len(mel.shape) == 2
        mel_infer_norm = self.wavegan_scaler.transform(mel.T).T
        mel_infer_norm = torch.from_numpy(mel_infer_norm).float().unsqueeze(0)
        with torch.no_grad():
            audio = self.wavegan.inference(mel_infer_norm, device=torch.device("cpu")).squeeze().numpy()
        audio = audio / max(abs(audio))
        return audio
