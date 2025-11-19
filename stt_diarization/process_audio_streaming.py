"""
Streaming audio processing module for real-time transcription and diarization.
Processes audio segments as they accumulate, rather than waiting for complete audio.
"""
import logging
import torch
import wave
import tempfile
from pathlib import Path
from speechbrain.inference.speaker import SpeakerRecognition
from config import (
    OUTPUT_DIR, SPEECHBRAIN_MODEL, DEVICE,
    SIMILARITY_THRESHOLD, MARGIN
)
from utils import load_and_resample
from pinecone_utils import get_pinecone_index, find_matching_speaker
from diarizer import load_diarizer, diarize_audio
from transcriber import load_whisper_model, transcribe_audio
from itertools import groupby

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global model instances (loaded once, reused)
_speaker_id_model = None
_diarization_pipeline = None
_whisper_model = None
_pinecone_index = None

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
        stats = _pinecone_index.describe_index_stats()
        if stats['namespaces'].get("reference", {}).get("vector_count", 0) == 0:
            logging.warning("No speaker embeddings enrolled in Pinecone 'reference' namespace.")
    
    return _speaker_id_model, _diarization_pipeline, _whisper_model, _pinecone_index

def identify_speaker(embedding_tensor, index, threshold=0.6):
    """Identify speaker by querying Pinecone with embedding vector."""
    return find_matching_speaker(index, embedding_tensor, threshold=threshold)

def save_audio_segment(audio_bytes, output_path):
    """Save audio bytes to a WAV file."""
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(16000)
        wf.writeframes(audio_bytes)
    return output_path

def process_audio_segment(audio_bytes, segment_offset=0.0):
    """
    Process an audio segment (bytes) through transcription and diarization.
    
    Parameters:
        audio_bytes: Raw PCM audio bytes (16kHz, mono, 16-bit)
        segment_offset: Time offset of this segment in the full recording (for timestamp alignment)
    
    Returns:
        str: The diarized transcript text (formatted as "Speaker: text" lines)
    """
    if not audio_bytes or len(audio_bytes) < 1600:  # Less than 0.05 seconds
        return ""
    
    # Save segment to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        save_audio_segment(audio_bytes, tmp_path)
        
        # Get models
        speaker_id_model, diarization_pipeline, whisper_model, index = get_models()
        
        # Load and resample audio
        meeting_waveform = load_and_resample(tmp_path)
        sample_rate = 16000
        
        # Run Whisper ASR
        transcription_result = transcribe_audio(whisper_model, tmp_path)
        word_list = [word for segment in transcription_result.get("segments", []) 
                     for word in segment.get("words", [])]
        
        if not word_list:
            return ""
        
        # Run speaker diarization
        diarization = diarize_audio(diarization_pipeline, tmp_path)
        
        speaker_map = {}
        speaker_turns = []
        
        # Identify speakers
        for segment, _, speaker_label in diarization.itertracks(yield_label=True):
            if speaker_label in speaker_map:
                identified_speaker = speaker_map[speaker_label]
            else:
                start_sample = int(segment.start * sample_rate)
                end_sample = int(segment.end * sample_rate)
                segment_waveform = meeting_waveform[:, start_sample:end_sample]
                
                if segment_waveform.shape[1] < 1600:
                    continue
                
                segment_embedding = speaker_id_model.encode_batch(segment_waveform).squeeze()
                identified_speaker, score = identify_speaker(segment_embedding, index, SIMILARITY_THRESHOLD)
                
                if not identified_speaker:
                    identified_speaker = f"Unknown_{speaker_label}"
                
                speaker_map[speaker_label] = identified_speaker
            
            # Adjust timestamps with segment offset
            speaker_turns.append({
                'start': segment.start + segment_offset,
                'end': segment.end + segment_offset,
                'speaker': identified_speaker
            })
        
        # Extract words aligned to speaker turns
        lines = []
        for turn in speaker_turns:
            turn_words = []
            start_time = turn["start"] - MARGIN
            end_time = turn["end"] + MARGIN
            
            for word in word_list:
                if word.get("start") is None or word.get("end") is None:
                    continue
                # Adjust word timestamps with segment offset
                word_start = word["start"] + segment_offset
                word_end = word["end"] + segment_offset
                if start_time <= word_start and word_end <= end_time:
                    turn_words.append(word["word"].strip())
            
            if turn_words:
                sentence = " ".join(turn_words).strip()
                lines.append((turn["start"], turn["speaker"], sentence))
        
        # Sort and merge lines
        lines.sort()
        merged_lines = []
        for speaker, group in groupby(lines, key=lambda x: x[1]):
            full_sentence = " ".join(g[2] for g in group)
            words = full_sentence.split()
            deduped = [words[0]] if words else []
            for i in range(1, len(words)):
                if words[i] != words[i - 1]:
                    deduped.append(words[i])
            merged_lines.append((speaker, " ".join(deduped)))
        
        # Build transcript text
        transcript_lines = []
        for speaker, text in merged_lines:
            transcript_lines.append(f"{speaker}: {text}")
        
        return "\n".join(transcript_lines)
        
    finally:
        # Clean up temporary file
        try:
            Path(tmp_path).unlink()
        except:
            pass

