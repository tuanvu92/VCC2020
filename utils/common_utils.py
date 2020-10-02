import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.decomposition import PCA


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none', cmap="Blues")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def get_list_of_files(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_list_of_files(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def read_hdf5(hdf5_name, hdf5_path):
    hdf5_file = h5py.File(hdf5_name, "r")
    if hdf5_path not in hdf5_file:
        print("There is no such a data in hdf5 file. ({hdf5_path})")
        return None
    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()
    return hdf5_data


def plot_speaker_emb(checkpoint_path, output_path):
    model_state_dict = torch.load(checkpoint_path)
    speaker_emb = model_state_dict["speaker_emb_layer.weight"].data.cpu().numpy().T
    pca = PCA(n_components=16)
    speaker_emb_pca = pca.fit_transform(speaker_emb)
    plt.figure(dpi=200, figsize=(16, 9))
    for i in range(1, 16):
        plt.subplot(3, 5, i)
        plt.scatter(speaker_emb_pca[:, 0], speaker_emb_pca[:, i])
        plt.annotate("TFF1", (speaker_emb_pca[2, 0], speaker_emb_pca[2, i]))
        plt.annotate("TFM1", (speaker_emb_pca[3, 0], speaker_emb_pca[3, i]))
    plt.savefig(output_path)


def read_file_list(fp):
    with open(fp, "r") as f:
        file_list = f.read().split("\n")
    file_list = [f for f in file_list if f != ""]
    return file_list
