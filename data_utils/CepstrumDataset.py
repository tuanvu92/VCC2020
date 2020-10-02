import torch
from torch.utils.data import Dataset
import random
import numpy as np
from utils.common_utils import read_file_list
from progressbar import progressbar


class MelCepstrumDataset(Dataset):
    def __init__(self, mel_file_list, use_f0=True, preload_data=False):
        if isinstance(mel_file_list, str):
            file_list = read_file_list(mel_file_list)
        elif isinstance(mel_file_list, list):
            file_list = mel_file_list
        else:
            print("Error: mel_file_list type is not supported: ", type(mel_file_list))
            return
        self.use_f0 = use_f0
        self.preload_data = preload_data
        self.mel_file_list = [fname for fname in file_list if fname.find(".npy") != -1]
        self.data = []
        if self.preload_data:
            print("Loading data to memory...")
            for fname in progressbar(self.mel_file_list):
                _mel = np.load(fname)
                mcc = torch.from_numpy(self.mel2mcc(_mel)).float()
                self.data.append(mcc)
        self.speaker_label = self.create_speaker_label()
        random.shuffle(self.mel_file_list)
        self.n_speaker = len(self.speaker_label)

    def __len__(self):
        return len(self.mel_file_list)

    @staticmethod
    def mel2mcc(mel):
        if len(mel.shape) == 2:
            mel = np.expand_dims(mel, 0)
        c = np.fft.irfft(mel, axis=1)
        c[:, 0] /= 2.0
        c = c[:, :mel.shape[1]]
        return np.squeeze(c)

    @staticmethod
    def mcc2mel(mcc):
        if len(mcc.shape) == 2:
            mcc = np.expand_dims(mcc, 0)
        sym_c = np.zeros([mcc.shape[0], 2 * (mcc.shape[1] - 1), mcc.shape[2]])
        sym_c[:, 0] = 2 * mcc[:, 0]
        for i in range(1, mcc.shape[1]):
            sym_c[:, i] = mcc[:, i]
            sym_c[:, -i] = mcc[:, i]
        mel = np.fft.rfft(sym_c, axis=1).real
        mel = np.squeeze(mel)
        return mel

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
        if self.preload_data:
            mcc = self.data[index]
        else:
            mel = np.load(self.mel_file_list[index])
            mcc = torch.from_numpy(self.mel2mcc(mel)).float()

        if self.use_f0:
            f0_file = self.mel_file_list[index].replace("mel24k", "f0_24k").replace("mel16k", "f0_16k")
            f0 = torch.from_numpy(np.load(f0_file)).float()
            f0 = f0[:mcc.shape[-1]]
            x = torch.cat([mcc, f0.unsqueeze(0)], dim=0)
        else:
            x = mcc
        y = torch.zeros([1, self.n_speaker])
        speaker_label = self.get_label(self.mel_file_list[index])
        y[0, speaker_label] = 1.0
        return x, y

    def __getitem__(self, index):
        return self.get_data(index)


class MelCepstrumCollateFn(object):
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
        padded_seq = torch.zeros([batch_size, feature_dim, batch_max_seq_len])

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


class MelCepstrumCollateFnNoTruncate(object):
    def __init__(self):
        return

    def __call__(self, batch):
        batch_size = len(batch)
        feature_dim = batch[0][0].shape[0]
        label = torch.cat([x[1] for x in batch], dim=0)
        seq_lengths = [x[0].shape[1] for x in batch]
        batch_max_seq_len = max(seq_lengths)
        padded_seq = torch.zeros([batch_size, feature_dim, batch_max_seq_len])
        for i, (seq, _) in enumerate(batch):
            if batch_max_seq_len < seq.shape[1]:
                # Select random segment from sequence
                start = torch.randint(low=0, high=seq.shape[1] - batch_max_seq_len, size=[1])
                padded_seq[i] = seq[:, start: start + batch_max_seq_len]
            else:
                # Wrap-padding sequence to batch_max_seq_len
                n = batch_max_seq_len // seq.shape[1]
                for j in range(n):
                    padded_seq[i, :, seq.shape[1] * j: seq.shape[1] * (j + 1)] = seq
                wrap_len = batch_max_seq_len % seq.shape[1]
                if wrap_len > 0:
                    padded_seq[i, :, -wrap_len:] = seq[:, :wrap_len]
        return padded_seq, label, seq_lengths