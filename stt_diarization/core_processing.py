import logging
import torch
import torchaudio
import uuid
from datetime import datetime
from pathlib import Path
from speechbrain.inference.speaker import SpeakerRecognition
from config import (
    OUTPUT_DIR, SPEECHBRAIN_MODEL, DEVICE,
    SIMILARITY_THRESHOLD, UNKNOWN_VOICES_DIR, INDEX_NAME
)
from pinecone_utils import get_pinecone_index, find_matching_speaker, upsert_embeddings
from diarizer import load_diarizer
from transcriber import load_whisper_model
from cloudinary_utils import upload_audio_to_cloudinary

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UNKNOWN_VOICES_DIR.mkdir(parents=True, exist_ok=True)

# Global model instances (loaded once, reused)
_speaker_id_model = None
_diarization_pipeline = None
_whisper_model = None
_pinecone_index = None
_local_speaker_cache = {}  # Local cache for recently enrolled speakers

def get_models():
    """Load and cache models (singleton pattern for efficiency)."""
    global _speaker_id_model, _diarization_pipeline, _whisper_model, _pinecone_index
    
    if _speaker_id_model is None:
        logging.info("Loading speaker recognition model...")
        project_dir = Path(__file__).parent
        pretrained_dir = project_dir / "pretrained_models" / SPEECHBRAIN_MODEL.replace('/', '_')
        _speaker_id_model = SpeakerRecognition.from_hparams(
            source=SPEECHBRAIN_MODEL,
            savedir=str(pretrained_dir),
            run_opts={"device": DEVICE}
        )
    
    if _diarization_pipeline is None:
        logging.info("Loading diarization pipeline...")
        _diarization_pipeline = load_diarizer()
    
    if _whisper_model is None:
        logging.info("Loading Whisper model...")
        _whisper_model = load_whisper_model()
    
    if _pinecone_index is None:
        logging.info("Loading Pinecone index...")
        _pinecone_index = get_pinecone_index()
        # Verify enrolled speakers exist
        stats = _pinecone_index.describe_index_stats()
        if stats['namespaces'].get("reference", {}).get("vector_count", 0) == 0:
            logging.warning("No speaker embeddings enrolled in Pinecone 'reference' namespace.")
    
    return _speaker_id_model, _diarization_pipeline, _whisper_model, _pinecone_index

def check_local_cache(embedding_tensor, threshold=0.6):
    """Check local cache for matching speaker."""
    best_score = -1.0
    best_speaker = None
    
    # cache_size = len(_local_speaker_cache)
    # logging.info(f"🔍 Checking local cache (size={cache_size})...")
    
    for speaker_id, cached_embedding in _local_speaker_cache.items():
        score = torch.nn.functional.cosine_similarity(embedding_tensor, cached_embedding, dim=0).item()
        if score > best_score:
            best_score = score
            best_speaker = speaker_id
            
    if best_score >= threshold:
        logging.info(f"✅ Local cache hit: {best_speaker} (score={best_score:.4f})")
        return best_speaker, best_score
    
    return None, best_score

def identify_speaker(embedding_tensor, index, threshold=0.6):
    """
    Identify speaker by querying Pinecone with embedding vector for best match.
    Returns (speaker_id, score) if match passes threshold, else (None, score).
    """
    return find_matching_speaker(index, embedding_tensor, threshold=threshold)

def enroll_unknown_speaker(segment_waveform, segment_embedding, sample_rate, index):
    """
    Enroll a new unknown speaker: generate name, save audio, upsert to Pinecone, and cache locally.
    """
    # Generate unique name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:6]
    identified_speaker = f"Spk_{timestamp}_{unique_id}"
    
    # Save audio segment (non-blocking failure)
    audio_url = None
    try:
        audio_filename = UNKNOWN_VOICES_DIR / f"{identified_speaker}.wav"
        torchaudio.save(str(audio_filename), segment_waveform.cpu(), sample_rate)
        
        # Upload to Cloudinary
        audio_url = upload_audio_to_cloudinary(audio_filename, identified_speaker)
        
    except Exception as e:
        logging.error(f"❌ Failed to save/upload audio for {identified_speaker}: {e}")
    
    # Upsert embedding to Pinecone
    vector = segment_embedding.cpu().numpy().tolist()
    metadata = {'audio_url': audio_url} if audio_url else {}
    upsert_embeddings(index, [{'id': identified_speaker, 'values': vector, 'metadata': metadata}])
    
    # Add to local cache
    _local_speaker_cache[identified_speaker] = segment_embedding
    
    logging.info(f"⬆️✨Enrolled new speaker: {identified_speaker} | Cache size: {len(_local_speaker_cache)}")
    
    return identified_speaker
