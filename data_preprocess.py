import numpy as np
from data_utils.CepstrumDataset import MelCepstrumDataset, MelCepstrumCollateFn
from sklearn.preprocessing import StandardScaler
from utils.stft import TacotronSTFT
from data_utils.AudioDataset import AudioDataset, AudioCollateFn
from torch.utils.data import DataLoader
from progressbar import progressbar
from os.path import join, exists
import math
import os
from utils.common_utils import get_list_of_files
import librosa
import pyworld
import json


def calculate_mcc_stats(file_list, output="stats_mcc.npy"):
    """ Calculate mean and variance of mcc of dataset
    Args:
        file_list (str): path to file list
        output (str): path to save output npy file

    """
    dataset = MelCepstrumDataset(mel_file_list=file_list)
    scaler = StandardScaler()
    for mel, _ in progressbar(dataset):
        scaler.partial_fit(mel.T)
    mean = scaler.mean_[np.newaxis, :]
    std = scaler.scale_[np.newaxis, :]
    stats = np.concatenate([mean, std], axis=0)
    np.save(output, stats)
    print("Save stats to ", output)


def extract_mel(wav_dir, fs, speaker_id_pos, filter_length, hop_length, mel_fmin, mel_fmax):
    """ Extract mel-spectrogram from audio waveform """
    mel_extractor = TacotronSTFT(filter_length=filter_length, hop_length=hop_length, n_mel_channels=80,
                                 sampling_rate=fs, mel_fmin=mel_fmin, mel_fmax=mel_fmax).cuda()
    audio_dataset = AudioDataset(wav_dir=wav_dir, n_speaker=None, speaker_id_pos=speaker_id_pos)
    audio_collate_fn = AudioCollateFn()
    audio_loader = DataLoader(audio_dataset,
                              batch_size=32,
                              shuffle=False,
                              collate_fn=audio_collate_fn,
                              num_workers=0)

    for batch in progressbar(audio_loader):
        audio, lengths, audio_path = batch
        audio = audio.cuda(0)
        mel = mel_extractor.mel_spectrogram(audio).detach().cpu().numpy()
        for i, input_wav in enumerate(audio_path):
            mel_fname = input_wav.split("/")[-1].replace(".wav", "")
            output_mel_dir = input_wav.replace("wav16", "mel16k")
            output_mel_dir = "/".join(output_mel_dir.split("/")[:-1])
            if not exists(output_mel_dir):
                os.makedirs(output_mel_dir)
            _mel = mel[i, :, :int(math.ceil(lengths[i]/hop_length))]
            np.save(join(output_mel_dir, mel_fname), _mel)


def file_filters(fname):
    ret = True
    if fname.find(".wav") == -1:
        ret = False
    if fname.find("falset10") != -1:
        ret = False
    if fname.find("whisper10") != -1:
        ret = False
    return ret


def extract_f0(wav_dir, speaker_id_pos=-4):
    wav_file_list = get_list_of_files(wav_dir)
    wav_file_list = [fname for fname in wav_file_list if file_filters(fname)]
    with open("jvs_speaker_info.json", "r") as f:
        speaker_info = json.load(f)
    for fname in progressbar(wav_file_list, redirect_stdout=True):
        print(fname)
        speaker_name = fname.split("/")[speaker_id_pos]
        x, fs = librosa.load(fname, sr=None)
        x = x.astype(np.float64)
        _f0, t = pyworld.dio(x, fs,
                             # f0_floor=75, f0_ceil=400,
                             f0_floor=speaker_info[speaker_name]["f0_min"],
                             f0_ceil=speaker_info[speaker_name]["f0_max"],
                             frame_period=12.5)
        f0 = pyworld.stonemask(x, _f0, t, fs)
        f0[f0 < 1.0] = 1.0
        f0 = np.log2(f0).astype(np.float32)
        fname = fname.replace("wav24", "f0_24k")
        fname = fname.replace(".wav", "")
        fname_tokens = fname.split('/')
        file_name = fname_tokens[-1]
        output_dir = "/".join(fname_tokens[:speaker_id_pos+1])
        if not exists(output_dir):
            os.makedirs(output_dir)
        np.save(join(output_dir, file_name), f0)
    print("Finished!")


if __name__ == "__main__":
    # calculate_mcc_stats("file_lists/vctk_file_list.txt", "stats/vctk_mcc_stats.npy")
    # extract_mel("/home/messier/PycharmProjects/data/VCTK/wav16/",
    #             fs=16000, speaker_id_pos=-2, filter_length=1024,
    #             hop_length=256, mel_fmin=80, mel_fmax=7600)
    extract_f0("/home/messier/PycharmProjects/data/jvs_ver1/wav24", speaker_id_pos=-4)
