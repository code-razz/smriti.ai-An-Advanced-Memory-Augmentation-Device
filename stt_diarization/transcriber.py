import whisper
from config import WHISPER_MODEL_NAME, DEVICE

def load_whisper_model():
    """
    Load Whisper ASR model of specified size.

    Returns:
        model (whisper.Whisper): Loaded Whisper model.
    """
    return whisper.load_model(WHISPER_MODEL_NAME, device=DEVICE)

def transcribe_audio(model, audio_input):
    """
    Generate transcription of audio with word-level timestamps.

    Parameters:
        model (whisper.Whisper): Loaded Whisper model.
        audio_input (str, Path, or np.ndarray): Path to audio file or numpy array of audio.

    Returns:
        dict: Transcription result containing segments with word-level timestamps.
    """
    # If audio_input is a path (str or Path), Whisper handles it.
    # If it's a numpy array, Whisper also handles it.
    # We just pass it through, but ensure paths are strings if they are Path objects.
    if hasattr(audio_input, 'resolve'): # Check if it's a Path object
        audio_input = str(audio_input)
        
    return model.transcribe(audio_input, word_timestamps=True)
