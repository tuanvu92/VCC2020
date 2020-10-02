# -*- coding: utf-8 -*-
""" Logger modules for tensorboard

Author: Ho Tuan Vu - Japan Advanced Institute of Science and Technology
Revision: 1.0

"""
from tensorboardX import SummaryWriter
from utils.common_utils import *


class DataLogger(SummaryWriter):
    """ Logger modules for tensorboard """
    def __init__(self, logdir):
        """ Initialize DataLogger module
        Args:
            logdir (str): Directory contains log file

        """
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
            os.chmod(logdir, 0o775)
        super().__init__(logdir)

    def log_training(self, loss_value, loss_name, iteration):
        for _value, _name in zip(loss_value, loss_name):
            if isinstance(_value, list):
                self.add_scalars(_name[0], {s: v for s, v in zip(_name[1:], _value)}, iteration)
            else:
                self.add_scalar(_name, _value, iteration)

    def log_validation(self, losses: dict,
                       figures: dict,
                       audio: dict,
                       fs=24000,
                       iteration=0):
        for _name, _value in losses.items():
            self.add_scalar(_name, _value, iteration)

        for _name, _fig in figures.items():
            self.add_figure(_name, _fig, iteration)

        for _name, _audio in audio.items():
            self.add_audio(_name, _audio, iteration, sample_rate=fs)
