import torch
from torch.utils.data import Dataset
import random
import numpy as np
from utils.common_utils import read_file_list
import pyworld


class MelSpectrumDataset(Dataset):
    def __init__(self, mel_file_list):
        file_list = read_file_list(mel_file_list)
        self.mel_file_list = [fname for fname in file_list if fname.find(".npy") != -1]
        self.speaker_label = self.create_speaker_label()
        random.shuffle(self.mel_file_list)
        self.n_speaker = len(self.speaker_label)

    def __len__(self):
        return len(self.mel_file_list)

    def create_speaker_label(self):
        speaker_list = []
        for fname in self.mel_file_list:
            speaker_name = fname.split("/")[-2]
            if speaker_name not in speaker_list:
                speaker_list.append(speaker_name)
        speaker_list.sort()
        speaker_label = {spkr_name: int(i) for spkr_name, i in zip(speaker_list, np.arange(len(speaker_list)))}
        return speaker_label

    def get_label(self, fname):
        speaker_name = fname.split("/")[-2]
        return self.speaker_label[speaker_name]

    def get_data(self, index):
        mel = torch.from_numpy(np.load(self.mel_file_list[index])).float()
        f0 = torch.from_numpy(np.load(self.mel_file_list[index]).replace("mel16k", "f0_16k")).float()
        x = torch.stack([mel, f0.unsqueeze(0)])
        y = torch.zeros([1, self.n_speaker])
        speaker_label = self.get_label(self.mel_file_list[index])
        y[0, speaker_label] = 1.0
        return x, y

    def __getitem__(self, index):
        return self.get_data(index)


class MelSpectrumCollateFn(object):
    def __init__(self, max_seq_len=512):
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        # batch = [seq, label]
        # seq.shape = [T,]
        # label.shape = [n_speaker]
        batch_size = len(batch)
        feature_dim = batch[0][0].shape[0]
        label = torch.cat([x[1] for x in batch], dim=0)
        seq_lengths = [x[0].shape[1] for x in batch]
        batch_max_seq_len = min(self.max_seq_len, max(seq_lengths))
        batch_max_seq_len = batch_max_seq_len - batch_max_seq_len % 8
        padded_seq = torch.zeros([batch_size, feature_dim,  batch_max_seq_len])

        for i, (seq, _) in enumerate(batch):
            if batch_max_seq_len < seq.shape[1]:
                # Select random segment from sequence
                start = torch.randint(low=0, high=seq.shape[1] - batch_max_seq_len, size=[1])
                padded_seq[i] = seq[:, start: start + batch_max_seq_len]
            else:
                # Wrap-padding sequence to batch_max_seq_len
                n = batch_max_seq_len // seq.shape[1]
                for j in range(n):
                    padded_seq[i, :, seq.shape[1]*j: seq.shape[1]*(j+1)] = seq
                wrap_len = batch_max_seq_len % seq.shape[1]
                if wrap_len > 0:
                    padded_seq[i, :, -wrap_len:] = seq[:, :wrap_len]

        return padded_seq, label
