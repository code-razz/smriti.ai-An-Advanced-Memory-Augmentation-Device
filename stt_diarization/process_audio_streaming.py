"""
Streaming audio processing module for real-time transcription and diarization.
Processes audio segments as they accumulate, rather than waiting for complete audio.
"""
import warnings
# Suppress torchaudio deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*")
warnings.filterwarnings("ignore", message=".*TorchAudio.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd.*")
warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom.*")
warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")
# Suppress SpeechBrain deprecation warnings
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*deprecated.*")
# Suppress pyannote warnings about symlinks on Windows
warnings.filterwarnings("ignore", message=".*Pretrainer collection using symlinks on Windows.*")

import logging
import wave
import tempfile
from pathlib import Path
from config import OUTPUT_DIR, SIMILARITY_THRESHOLD, MARGIN
from utils import load_and_resample
from diarizer import diarize_audio
from transcriber import transcribe_audio
from itertools import groupby
from core_processing import (
    get_models, check_local_cache, identify_speaker, enroll_unknown_speaker
)

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
                
                min_length = int(1.0 * sample_rate)  # 1 second in samples
                if segment_waveform.shape[1] < min_length:
                    continue

                segment_embedding = speaker_id_model.encode_batch(segment_waveform).squeeze()
                
                # 1. Check local cache first
                identified_speaker, score = check_local_cache(segment_embedding, SIMILARITY_THRESHOLD)
                
                # 2. If not in cache, query Pinecone
                if not identified_speaker:
                    identified_speaker, score = identify_speaker(segment_embedding, index, SIMILARITY_THRESHOLD)
                
                if not identified_speaker:
                    identified_speaker = enroll_unknown_speaker(segment_waveform, segment_embedding, sample_rate, index)
                
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
