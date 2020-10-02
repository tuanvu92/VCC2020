import torch
from torch.utils.data import Dataset
from utils.common_utils import get_list_of_files
import librosa
import numpy as np


class AudioDataset(Dataset):
    def __init__(self, wav_dir, n_speaker=None, speaker_id_pos=-2):
        wav_file_list = get_list_of_files(wav_dir)
        self.wav_file_list = [s for s in wav_file_list if self.file_filters(s)]
        self.speaker_label = self.create_speaker_label(n_speaker=n_speaker,
                                                       speaker_id_pos=speaker_id_pos)
        self.wav_file_list = [fname for fname in self.wav_file_list
                              if fname.split("/")[speaker_id_pos] in self.speaker_label]
        self.wav_file_list.sort()
        self.n_speaker = n_speaker

    @staticmethod
    def file_filters(fname):
        ret = True
        if fname.find(".wav") == -1:
            ret = False
        if fname.find("falset10") != -1:
            ret = False
        if fname.find("whisper10") != -1:
            ret = False
        return ret

    def get_audio(self, idx):
        x, fs = librosa.load(self.wav_file_list[idx], sr=None)
        try:
            x = x / max(abs(x))
        except ValueError:
            print(self.wav_file_list[idx])
            print(len(x))
            print(abs(x))

        return torch.from_numpy(x).type(torch.float)

    def get_label(self, fname):
        speaker_name = fname.split("/")[-1].split("_")[0]
        return self.speaker_label[speaker_name]

    def create_speaker_label(self, n_speaker=None, speaker_id_pos=-2):
        speaker_list = []
        for fname in self.wav_file_list:
            speaker_name = fname.split("/")[speaker_id_pos]
            if speaker_name not in speaker_list:
                speaker_list.append(speaker_name)
        speaker_list.sort()
        if n_speaker is not None:
            speaker_list = speaker_list[:n_speaker]
        speaker_label = {spkr_name: i for spkr_name, i in zip(speaker_list, np.arange(len(speaker_list)))}
        return speaker_label

    def __getitem__(self, index):
        if self.n_speaker is not None:
            y = torch.zeros([1, self.n_speaker])
            speaker_label = self.get_label(self.wav_file_list[index])
            y[0, speaker_label] = 1.0
            return self.get_audio(index), self.wav_file_list[index], y
        else:
            return self.get_audio(index), self.wav_file_list[index], torch.tensor([0, 0])

    def __len__(self):
        return len(self.wav_file_list)

    @staticmethod
    def mel2mcc(mel):
        c = np.fft.irfft(mel, axis=1)
        c[:, 0, :] /= 2.0
        return c[:, :mel.shape[1], :]

    @staticmethod
    def mcc2mel(mcc):
        sym_c = np.zeros([mcc.shape[0], 2*(mcc.shape[1]-1), mcc.shape[2]])
        sym_c[:, 0, :] *= 2.0
        for i in range(1, mcc.shape[1]):
            sym_c[:, i, :] = mcc[:, i, :]
            sym_c[:, -i, :] = mcc[:, i, :]

        mel = np.fft.rfft(sym_c, axis=1).real
        return mel


class AudioCollateFn(object):
    def __init__(self, onehot=False):
        self.onehot = onehot
        return

    def __call__(self, batch):
        batch_size = len(batch)
        audio_path = [_item[1] for _item in batch]
        audio_len = [x[0].shape[0] for x in batch]

        y = torch.cat([x[2] for x in batch], dim=0)

        max_audio_len = max(audio_len)
        # if max_audio_len < 256*512:
        #     max_audio_len = 256*512
        # max_audio_len = 8*256*(max_audio_len//(8*256) + 1)
        padded_audio = torch.zeros([batch_size, max_audio_len])

        for i, x in enumerate(batch):
            padded_audio[i, :x[0].shape[0]] = x[0]
            audio_len[i] = x[0].shape[0]
        if self.onehot:
            return padded_audio, audio_len, audio_path, y
        else:
            return padded_audio, audio_len, audio_path
