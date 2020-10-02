import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.common_utils import get_list_of_files
from progressbar import progressbar
import json
from os.path import join


def calc_f0_stats(file_list):
    scaler = StandardScaler()
    for fname in progressbar(file_list):
        f0 = np.load(fname)
        f0 = f0[f0 > 1]
        if len(f0) > 1:
            scaler.partial_fit(f0.reshape(-1, 1))
    return float(scaler.mean_.squeeze()), float(scaler.scale_.squeeze())


def calc_f0_stats_dataset(data_dir, output_file="vctk_f0_stats.json"):
    file_list = get_list_of_files(data_dir)
    file_list = [fname for fname in file_list if fname.find(".npy") != -1]
    speaker_list = []
    for fname in file_list:
        speaker_name = fname.split("/")[-2]
        if speaker_name not in speaker_list:
            speaker_list.append(speaker_name)
    speaker_f0_stats = dict()
    for speaker_name in speaker_list:
        print(speaker_name)
        file_list_speaker = [fname for fname in file_list if fname.find(speaker_name) != -1]
        mean, std = calc_f0_stats(file_list_speaker)
        speaker_f0_stats[speaker_name] = [mean, std]

    with open(output_file, "w") as f:
        json.dump(speaker_f0_stats, f, indent=1)


if __name__ == "__main__":
    calc_f0_stats_dataset("/home/messier/PycharmProjects/data/vcc2020_training/f0_24k",
                          output_file="../vcc2020_f0_24k_stats.json")
