import torch
from pyannote.audio import Pipeline
from config import PYANNOTE_PIPELINE, HUGGINGFACE_TOKEN, DEVICE

def load_diarizer():
    """
    Load pretrained Pyannote speaker diarization pipeline.

    Returns:
        pipeline (pyannote.audio.Pipeline): Initialized diarization pipeline on proper device.
    """
    pipeline = Pipeline.from_pretrained(PYANNOTE_PIPELINE, use_auth_token=HUGGINGFACE_TOKEN)
    pipeline.to(torch.device(DEVICE))
    return pipeline

def diarize_audio(pipeline, audio_file):
    """
    Run speaker diarization on audio file.

    Parameters:
        pipeline (pyannote.audio.Pipeline): Preloaded diarization pipeline.
        audio_file (str or Path): Path to audio file.

    Returns:
        diarization (Annotation): Pyannote diarization output with segments and speaker labels.
    """
    return pipeline(audio_file)
