import whisper
from config import WHISPER_MODEL_NAME, DEVICE

def load_whisper_model():
    """
    Load Whisper ASR model of specified size.

    Returns:
        model (whisper.Whisper): Loaded Whisper model.
    """
    return whisper.load_model(WHISPER_MODEL_NAME, device=DEVICE)

def transcribe_audio(model, audio_path):
    """
    Generate transcription of audio with word-level timestamps.

    Parameters:
        model (whisper.Whisper): Loaded Whisper model.
        audio_path (str or Path): Path to audio file.

    Returns:
        dict: Transcription result containing segments with word-level timestamps.
    """
    return model.transcribe(str(audio_path), word_timestamps=True)
