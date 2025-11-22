"""
Helper module for processing audio files through diarization and transcription.
Can be imported and used by server.py for real-time processing.
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

def process_audio_file(audio_file_path, output_file_path=None):
    """
    Process an audio file through transcription and diarization.
    
    Parameters:
        audio_file_path (str or Path): Path to the audio file to process
        output_file_path (str or Path, optional): Path to save the output transcript.
            If None, saves to OUTPUT_DIR/named_diarized_output.txt
    
    Returns:
        str: The diarized transcript text (formatted as "Speaker: text" lines)
    """
    audio_file_path = Path(audio_file_path)
    if not audio_file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    
    if output_file_path is None:
        output_file_path = OUTPUT_DIR / "named_diarized_output.txt"
    else:
        output_file_path = Path(output_file_path)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Processing audio file: {audio_file_path}")
    
    # Get models (will load on first call, then reuse)
    speaker_id_model, diarization_pipeline, whisper_model, index = get_models()
    
    # Load meeting audio and resample
    meeting_waveform = load_and_resample(audio_file_path)
    sample_rate = 16000  # Fixed target sample rate
    
    # Run Whisper ASR for transcription with word timestamps
    logging.info("Running Whisper transcription...")
    transcription_result = transcribe_audio(whisper_model, str(audio_file_path))
    word_list = [word for segment in transcription_result.get("segments", []) 
                 for word in segment.get("words", [])]
    
    # Run speaker diarization
    logging.info("Running speaker diarization...")
    diarization = diarize_audio(diarization_pipeline, str(audio_file_path))
    
    speaker_map = {}      # Map from diarizer speaker label to enrolled speaker ID or Unknown label
    speaker_turns = []    # Collect speaker-labeled time segments
    
    # Iterate diarization segments and identify speakers
    logging.info("Identifying speakers...")
    for segment, _, speaker_label in diarization.itertracks(yield_label=True):
        if speaker_label in speaker_map:
            identified_speaker = speaker_map[speaker_label]
        else:
            # Extract segment waveform using sample-level indices
            start_sample = int(segment.start * sample_rate)
            end_sample = int(segment.end * sample_rate)
            segment_waveform = meeting_waveform[:, start_sample:end_sample]
            
            # Skip segments that are too short for reliable embedding
            if segment_waveform.shape[1] < 1600:
                continue
            
            # Compute embedding and identify speaker via Pinecone query
            segment_embedding = speaker_id_model.encode_batch(segment_waveform).squeeze()
            
            # 1. Check local cache first
            identified_speaker, score = check_local_cache(segment_embedding, SIMILARITY_THRESHOLD)
            
            # 2. If not in cache, query Pinecone
            if not identified_speaker:
                identified_speaker, score = identify_speaker(segment_embedding, index, SIMILARITY_THRESHOLD)
            
            # Assign unknown speaker if confidence low
            if not identified_speaker:
                identified_speaker = enroll_unknown_speaker(segment_waveform, segment_embedding, sample_rate, index)
            
            speaker_map[speaker_label] = identified_speaker
        
        speaker_turns.append({'start': segment.start, 'end': segment.end, 'speaker': identified_speaker})
    
    # Extract words aligned to speaker turns within margin, build transcript lines
    logging.info("Aligning words to speaker turns...")
    lines = []
    for turn in speaker_turns:
        turn_words = []
        start_time = turn["start"] - MARGIN
        end_time = turn["end"] + MARGIN
        
        for word in word_list:
            if word.get("start") is None or word.get("end") is None:
                continue
            if start_time <= word["start"] and word["end"] <= end_time:
                turn_words.append(word["word"].strip())
        
        if turn_words:
            sentence = " ".join(turn_words).strip()
            lines.append((turn["start"], turn["speaker"], sentence))
    
    # Sort lines by time
    lines.sort()
    
    # Merge continuous speech from same speaker and remove redundant repeated words
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
        output_line = f"{speaker}: {text}"
        transcript_lines.append(output_line)
    
    transcript_text = "\n".join(transcript_lines)
    
    # Write final transcript to output file
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    
    logging.info(f"Transcript saved to {output_file_path}")
    logging.info(f"Processed {len(merged_lines)} speaker turns")
    
    return transcript_text
