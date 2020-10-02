# -*- coding: utf-8 -*-
""" Main training script

Author: Ho Tuan Vu - Japan Advanced Institute of Science and Technology
Revision: 1.0

"""

import matplotlib
matplotlib.use("Agg")
from os.path import join, exists
import json
import argparse
import subprocess
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from utils.logger import DataLogger
from utils.common_utils import *
from time import localtime, strftime
from data_utils.MelSpectrumDataset import MelSpectrumDataset, MelSpectrumCollateFn
from data_utils.CepstrumDataset import MelCepstrumDataset, MelCepstrumCollateFn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from models.vqvae_3stage import VQVAE3Stage
from models.vqvae_2stage import VQVAE2Stage
from models.vqvae_1stage import VQVAE1Stage
from torch.optim import Adam
from progressbar import progressbar
from utils.synthesizer import PWGSynthesizer
import librosa


def train(model_name, train_list, max_seq_len, batch_size, train_epoch,
          learning_rate, iters_per_checkpoint, iters_per_eval,
          n_warm_up_epoch, warm_up_lr, checkpoint_dir, use_f0=True, preload_data=False,
          checkpoint_path="", seed=12345, num_gpus=1, rank=0, group_name=""):
    torch.manual_seed(seed)
    if num_gpus > 1:
        init_distributed(rank=rank, num_gpus=num_gpus, group_name=group_name, **dist_configs)

    timestamp = strftime("%Y%m%d_%H%M_" + checkpoint_dir, localtime())
    output_path = join("checkpoints/", timestamp)
    dataset = MelCepstrumDataset(train_list, use_f0=use_f0, preload_data=preload_data)

    if rank == 0:
        print("Checkpoint dir: %s" % output_path)
        if not exists(output_path):
            os.makedirs(output_path)
        subprocess.run(["cp", "-r", args.config, "modules", "models", output_path])
        with open(join(output_path, "speaker_label.json"), "w") as f:
            json.dump(dataset.speaker_label, f)

    train_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
    print("Data directory: ", train_list)
    print("No. training data: ", len(dataset))
    print("No. speakers:", dataset.n_speaker)
    print("Normalize: ", model_configs["norm"])
    print("Use F0: ", use_f0)
    collate_fn = MelCepstrumCollateFn(max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset=dataset,
                            sampler=train_sampler,
                            batch_size=batch_size//num_gpus,
                            collate_fn=collate_fn,
                            num_workers=4,
                            pin_memory=True,
                            shuffle=False)
    model = None
    if model_name == "VQVAE3Stage":
        model = VQVAE3Stage(n_speaker=dataset.n_speaker, **model_configs).cuda()
    elif model_name == "VQVAE2Stage":
        model = VQVAE2Stage(n_speaker=dataset.n_speaker, **model_configs).cuda()
    elif model_name == "VQVAE1Stage":
        model = VQVAE1Stage(n_speaker=dataset.n_speaker, **model_configs).cuda()
    else:
        print("Unsupported model name: %s" % model_name)
    if checkpoint_path != "":
        print(checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))

    # =====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)
    # =====END:   ADDED FOR DISTRIBUTED======

    optimizer = Adam(model.parameters(), lr=warm_up_lr)
    if rank == 0:
        logger = DataLogger(logdir=join(output_path, "logs"))
        validator = Validator(logger=logger,
                              speaker_label=dataset.speaker_label,
                              use_f0=use_f0,
                              **validation_configs)

    else:
        logger = None
        validator = None
    iteration = 0
    for epoch in range(train_epoch):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if rank == 0:
            iterator = progressbar(dataloader, redirect_stdout=True)
        else:
            iterator = dataloader

        for batch in iterator:
            model.zero_grad()
            batch = [batch[0].cuda(), batch[1].cuda()]
            loss, loss_components = model(batch)

            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
                for i in range(len(loss_components)):
                    if isinstance(loss_components[i], list):
                        for j in range(len(loss_components[i])):
                            loss_components[i][j] = reduce_tensor(loss_components[i][j].data, num_gpus).item()
                    else:
                        loss_components[i] = reduce_tensor(loss_components[i].data, num_gpus).item()
            else:
                reduced_loss = loss.item()
                for i in range(len(loss_components)):
                    if isinstance(loss_components[i], list):
                        for j in range(len(loss_components[i])):
                            loss_components[i][j] = loss_components[i][j].item()
                    else:
                        loss_components[i] = loss_components[i].item()
            loss.backward()
            optimizer.step()
            if rank == 0:
                rc_loss, mel_loss, vq_loss, commitment_loss, perplexity = loss_components
                print("%d|%d: loss=%.2e, rc_loss=%.2e, mel_loss=%.2e, vq_loss=%.2e" %
                      (epoch, iteration, reduced_loss, rc_loss, mel_loss, vq_loss))
                perplexity_tag = ["training/perplexity"] + [str(i) for i in range(len(perplexity))]

                if logger is not None:
                    logger.log_training([reduced_loss, rc_loss, mel_loss, vq_loss, perplexity],
                                        ["training/loss", "training/rc_loss", "training/mel_loss",
                                         "training/vq_loss", perplexity_tag],
                                        iteration)

                if (iteration % iters_per_eval) == 0:
                    torch.save(model.state_dict(), join(output_path, "weight_latest.pt"))
                    if validator is not None:
                        validator(model, iteration)

                if (iteration % iters_per_checkpoint) == 0 and iteration > 0:
                    torch.save(model.state_dict(),
                               join(output_path, "weight_%d.pt" % iteration))
            iteration += 1
        if epoch < n_warm_up_epoch:
            lr = min(learning_rate,
                     warm_up_lr - epoch * (warm_up_lr - learning_rate)/n_warm_up_epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    print("Finished!")
    return


class Validator(object):
    def __init__(self, logger: DataLogger, speaker_label,
                 test_mel_file, test_f0_file, target_speaker,
                 eval_list, max_seq_len, f0_stats_file, use_f0,
                 fs, synthesizer_configs, speaker_info_json=None):
        self.fs = fs
        eval_dataset = MelCepstrumDataset(eval_list, use_f0=use_f0)
        eval_dataset.speaker_label = speaker_label
        eval_dataset.n_speaker = len(speaker_label)

        collate_fn = MelCepstrumCollateFn(max_seq_len=max_seq_len)
        self.eval_dataloader = DataLoader(eval_dataset,
                                          collate_fn=collate_fn,
                                          batch_size=32,
                                          shuffle=False)
        source_speaker = test_mel_file.split("/")[-2]
        self.mel_src = np.load(test_mel_file)
        self.mel_tar = np.load(test_mel_file.replace(source_speaker, target_speaker))

        src_len = self.mel_src.shape[-1]
        self.mel_src = self.mel_src[:, :(src_len - src_len % 8)]
        self.mcc_src = torch.from_numpy(eval_dataset.mel2mcc(self.mel_src)).float()
        self.f0_src = np.load(test_f0_file)
        self.f0_src = self.f0_src[:8 * (self.f0_src.shape[0]//8)]

        with open(f0_stats_file, "r") as f:
            f0_stats = json.load(f)

        src_mean_f0, src_scale_f0 = f0_stats[source_speaker]
        tar_mean_f0, tar_scale_f0 = f0_stats[target_speaker]
        vuv = np.zeros(self.f0_src.shape[0])
        vuv[self.f0_src > 0] = 1.
        self.f0_tar = vuv * ((self.f0_src - src_mean_f0) * tar_scale_f0 / src_scale_f0 + tar_mean_f0)
        self.f0_tar = torch.from_numpy(self.f0_tar).float().unsqueeze(0)
        if use_f0:
            self.mcc_src = torch.cat([self.mcc_src, self.f0_tar], dim=0)
        self.mcc_src = self.mcc_src.unsqueeze(0).cuda()
        print("mcc_src shape: ", self.mcc_src.shape)
        self.target_id = speaker_label[target_speaker]
        self.logger = logger

        mel_src_fig = plt.figure(dpi=100, figsize=(9, 3))
        plt.imshow(self.mel_src, aspect='auto', origin='lower', cmap='Blues')
        plt.colorbar()
        plt.title("Converted")
        logger.add_figure("Validation/source", mel_src_fig, 0)
        plt.close()

        self.synthesizer = PWGSynthesizer(**synthesizer_configs)
        audio_src = self.synthesizer.synthesize(self.mel_src)
        audio_tar = self.synthesizer.synthesize(self.mel_tar)

        logger.add_audio("Source", audio_src, sample_rate=self.fs)
        logger.add_audio("Target", audio_tar, sample_rate=self.fs)

        if speaker_info_json is not None:
            with open(speaker_info_json, "r") as f:
                speaker_info = json.load(f)
            self.speaker_info = speaker_info
        else:
            self.speaker_info = None

    def __call__(self, model, iteration):
        rc_loss = []
        mel_loss = []
        vq_loss = []
        model.eval()
        with torch.no_grad():
            for batch in progressbar(self.eval_dataloader):
                model.zero_grad()
                batch = [batch[0].cuda(), batch[1].cuda()]

                _, loss_components = model(batch)
                for i in range(len(loss_components)):
                    if isinstance(loss_components[i], list):
                        continue
                    else:
                        loss_components[i] = loss_components[i].item()
                _rc_loss, _mel_loss, _vq_loss, _, _ = loss_components
                rc_loss.append(_rc_loss)
                mel_loss.append(_mel_loss)
                vq_loss.append(_vq_loss)
        rc_loss = np.mean(rc_loss)
        mel_loss = np.mean(mel_loss)
        vq_loss = np.mean(vq_loss)
        model.eval()
        speaker_id = torch.zeros([1, model.n_speaker])
        speaker_id[0, self.target_id] = 1.0
        with torch.no_grad():
            mel_conv = model.inference([self.mcc_src,
                                        speaker_id.cuda()]).squeeze().cpu().numpy()
        audio_conv = self.synthesizer.synthesize(mel_conv)
        emb = model.get_speaker_emb().cpu().numpy()
        pca_emb = PCA(n_components=2)
        emb_pca = pca_emb.fit_transform(emb)
        emb_fig = plt.figure(dpi=150)

        if self.speaker_info is None:
            plt.scatter(emb_pca[:, 0], emb_pca[:, 1], alpha=0.8)
        else:
            speaker_list = list(self.speaker_info.keys())
            idx_female = [i for i in range(len(speaker_list))
                          if self.speaker_info[speaker_list[i]]["gender"] == "F"]
            idx_male = [i for i in range(len(speaker_list))
                        if self.speaker_info[speaker_list[i]]["gender"] == "M"]
            plt.scatter(emb_pca[idx_female, 0], emb_pca[idx_female, 1], alpha=0.8, label="Female", marker="^")
            plt.scatter(emb_pca[idx_male, 0], emb_pca[idx_male, 1], alpha=0.8, label="Male", marker="v")
        plt.legend()
        plt.grid(linestyle="--")

        mel_fig = plt.figure(figsize=(9, 3), dpi=100)
        plt.imshow(mel_conv, aspect='auto', origin='lower', cmap='Blues')
        plt.colorbar()
        plt.title("Converted")
        self.logger.log_validation({"validation/rc_loss": rc_loss,
                                    "validation/mel_loss": mel_loss,
                                    "validation/vq_loss": vq_loss},
                                   {"validation/synthesized": mel_fig,
                                    "validation/speaker_embedding": emb_fig},
                                   {"converted": audio_conv},
                                   fs=self.fs,
                                   iteration=iteration)
        plt.close()
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='JSON file for configuration')
    global args
    args = parser.parse_args()
    num_gpus = torch.cuda.device_count()
    with open(args.config) as f:
        config = json.load(f)
    training_configs = config["training_configs"]

    global dist_configs
    dist_configs = config["dist_configs"]
    global model_configs
    model_configs = config["model_configs"]
    global validation_configs
    validation_configs = config["validation_configs"]

    if num_gpus > 1:
        if args.group_name == '':
            print("Warning: Training on 1 GPU!")
            num_gpus = 1
        else:
            print("Run distributed training on %d GPUs" % num_gpus)
    train(num_gpus=num_gpus, rank=args.rank, group_name=args.group_name, **training_configs)
