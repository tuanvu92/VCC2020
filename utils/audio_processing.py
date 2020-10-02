import torch
import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util


def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


def griffin_lim(magnitudes, stft_fn, n_iters=30):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log10(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return (10**x) / C


def mel2mcc(mel):
    if len(mel.shape) == 2:
        mel = mel.unsqueeze(0)
    N = mel.shape[1]
    mel = mel.transpose(1, 2).unsqueeze(-1)
    mel = torch.cat([mel, torch.zeros_like(mel)], dim=-1)
    mcc = torch.irfft(mel, signal_ndim=1, signal_sizes=[2 * (N - 1)]).transpose(1, 2)[:, :N]
    mcc[:, 0] /= 2.
    return mcc.squeeze()


def mel2mcc_np(mel):
    if len(mel.shape) == 2:
        mel = np.expand_dims(mel, 0)
    c = np.fft.irfft(mel, axis=1)
    c[:, 0] /= 2.0
    c = c[:, :mel.shape[1]]
    return np.squeeze(c)


def mcc2mel(mcc):
    if len(mcc.shape) == 2:
        mcc = mcc.unsqueeze(0)
    mcc[:, 0] *= 2.
    mcc = mcc.transpose(1, 2)
    mcc = torch.cat([mcc, torch.flip(mcc[:, :, 1:-1], dims=[-1])], dim=-1)
    mel = torch.rfft(mcc, signal_ndim=1)[:, :, :, 0]
    return mel.transpose(1, 2).squeeze()


def np_mel2mcc(mel):
    c = np.fft.irfft(mel, axis=0)
    c[0] /= 2.0
    return c[:mel.shape[0]]


def np_mcc2mel(mcc):
    sym_c = np.zeros([2*(mcc.shape[0]-1), mcc.shape[1]])
    sym_c[0] = 2*mcc[0]
    for i in range(1, mcc.shape[0]):
        sym_c[i] = mcc[i]
        sym_c[-i] = mcc[i]

    mel = np.fft.rfft(sym_c, axis=0).real
    return mel


if __name__ == "__main__":
    mel = np.load("../data/VCTK/mel24k/p225/p225_001.npy")
    mel = torch.from_numpy(mel).float()
    mcc = mel2mcc(mel)
    print(mcc.shape)
    mel_rc = mcc2mel(mcc)
    print(mel_rc.shape)
    err = (mel_rc - mel)**2
    print(err.numpy().mean())

    mcc_np = np_mel2mcc(mel.numpy())
    mel_np_rc = np_mcc2mel(mcc_np)

    err = (mcc.numpy() - mcc_np)**2
    print(err.mean())
