import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (like tokens, API keys)
load_dotenv()

# Project directories and paths
PROJECT_DIR = Path(__file__).parent.resolve()
REFERENCE_VOICES_DIR = PROJECT_DIR / "reference_voices"   # Reference speaker .wav files folder
OUTPUT_DIR = PROJECT_DIR / "output"                      # Output folder for transcripts
UNKNOWN_VOICES_DIR = PROJECT_DIR / "unknown_voices"      # Folder for unknown speaker audio segments
MEETING_AUDIO_FILE = PROJECT_DIR / "zapp_3_5_20.wav"   # Audio file to diarize and transcribe (update accordingly)
# MEETING_AUDIO_FILE = PROJECT_DIR / "videoplayback (1) (mp3cut.net).wav"   # Audio file to diarize and transcribe (update accordingly)
# MEETING_AUDIO_FILE = PROJECT_DIR / "videoplayback (1).wav"   # Audio file to diarize and transcribe (update accordingly)

# Model and pipeline names for loading pretrained models
WHISPER_MODEL_NAME = "base"
PYANNOTE_PIPELINE = "pyannote/speaker-diarization-3.1"
SPEECHBRAIN_MODEL = "speechbrain/spkrec-ecapa-voxceleb"  # Speaker recognition model

# Tokens & API keys for authentication and services
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone index and embedding dimensionality for speaker embeddings
INDEX_NAME = "speaker-embeddings"
EMBEDDING_DIM = 192  # ECAPA dimensions often 192 or 512

# Threshold for speaker similarity to consider a confident match
SIMILARITY_THRESHOLD = 0.60

# Margin time in seconds for extracting words around speaker turns in transcription
MARGIN = 0.25

# Select device based on CUDA availability or environment variables
# Select device based on CUDA availability or environment variables
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ STT Config: Using device: {DEVICE}")

# Cloudinary Credentials
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
