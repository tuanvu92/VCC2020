# -*- coding: utf-8 -*-
""" Inference script

Author: Ho Tuan Vu
Revision 1.0

"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import argparse
from models.vqvae_1stage import VQVAE1Stage
from models.vqvae_2stage import VQVAE2Stage
from models.vqvae_3stage import VQVAE3Stage

from utils.dsp import mel2mcc, mcc2mel
from utils.synthesizer import PWGSynthesizer
import librosa
import json
from os.path import join, exists
import matplotlib.pyplot as plt


def inference(model_name, model_checkpoint, model_configs, fs,
              synthesizer_configs, conversion_pairs_file, output_dir):
    """ Voice conversion
    Args:
        model_name (str): Name of conversion model
        model_checkpoint (str): path to checkpoint file
        model_configs (dict): model configuration dict
        synthesizer_configs (dict): synthesizer configuration dict
        conversion_pairs_file (str): file contains conversion pairs
        n_speaker (int): total speaker of trained model
        output_dir (str): path to output directory

    """
    if model_name == "VQVAE3Stage":
        model = VQVAE3Stage(**model_configs).eval().cuda()
    elif model_name == "VQVAE2Stage":
        model = VQVAE2Stage(**model_configs).eval().cuda()
    elif model_name == "VQVAE1Stage":
        model = VQVAE1Stage(**model_configs).eval().cuda()
    else:
        print("Unsupported model: ", model_name)
        exit()
    print("Model name: ", model_name)
    print("Output dir: ", output_dir)

    synthesizer = PWGSynthesizer(**synthesizer_configs)
    model.load_state_dict(torch.load(model_checkpoint))

    with open(conversion_pairs_file, "r") as f:
        conversion_pairs = f.read().split("\r")
    conversion_pairs = [s.split(" ") for s in conversion_pairs if s != ""]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='JSON file for configuration')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    inference(**config)
