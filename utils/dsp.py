import torch


def mel2mcc(mel):
    N = mel.shape[1]
    mel = mel.transpose(1, 2).unsqueeze(-1)
    mel = torch.cat([mel, torch.zeros_like(mel)], dim=-1)
    mcc = torch.irfft(mel, signal_ndim=1, signal_sizes=(2 * (N - 1),)).transpose(1, 2)[:, :N]
    mcc[:, 0] /= 2.
    return mcc


def mcc2mel(mcc):
    mcc[:, 0] *= 2.
    mcc = mcc.transpose(1, 2)
    mcc = torch.cat([mcc, torch.flip(mcc[:, :, 1:-1], dims=[-1])], dim=-1)
    mel = torch.rfft(mcc, signal_ndim=1)[:, :, :, 0]
    return mel.transpose(1, 2)
