import torch
import torchaudio

def load_and_resample(audio_path, target_sr=16000):
    """
    Load audio file and resample to the target sample rate.

    Parameters:
        audio_path (str or Path): Path to the audio file.
        target_sr (int): Desired sampling rate (default 16kHz).

    Returns:
        waveform (torch.Tensor): Mono audio tensor resampled to target_sr.
    """
    # Load waveform and original sample rate
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono by averaging channels if multi-channel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if sample rate does not match target sample rate
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform
