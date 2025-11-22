import logging
import torch
import torchaudio
import uuid
from datetime import datetime
from speechbrain.inference.speaker import SpeakerRecognition
from config import (
    MEETING_AUDIO_FILE, OUTPUT_DIR, SPEECHBRAIN_MODEL, DEVICE,
    SIMILARITY_THRESHOLD, MARGIN
)
from utils import load_and_resample
from pinecone_utils import get_pinecone_index, find_matching_speaker, upsert_embeddings
from diarizer import load_diarizer, diarize_audio
from transcriber import load_whisper_model, transcribe_audio
from itertools import groupby

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "named_diarized_output.txt"
UNKNOWN_VOICES_DIR = OUTPUT_DIR.parent / "unknown_voices"
UNKNOWN_VOICES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def identify_speaker(embedding_tensor, index, threshold=0.6):
    """
    Identify speaker by querying Pinecone with embedding vector for best match.
    Returns (speaker_id, score) if match passes threshold, else (None, score).
    """
    return find_matching_speaker(index, embedding_tensor, threshold=threshold)

def main():
    logging.info("Starting speaker diarization and transcription pipeline...")

    # Load Pinecone index and verify enrolled speakers exist
    index = get_pinecone_index()
    stats = index.describe_index_stats()
    if stats['namespaces'].get("reference", {}).get("vector_count", 0) == 0:
        raise RuntimeError("No speaker embeddings enrolled in Pinecone 'reference' namespace. Please run enroll_speakers.py.")

    # Load speaker recognition, diarization, and ASR models
    speaker_id_model = SpeakerRecognition.from_hparams(
        source=SPEECHBRAIN_MODEL,
        savedir=f"pretrained_models/{SPEECHBRAIN_MODEL.replace('/', '_')}",
        run_opts={"device": DEVICE}
    )
    diarization_pipeline = load_diarizer()
    whisper_model = load_whisper_model()

    # Load meeting audio and resample
    meeting_waveform = load_and_resample(MEETING_AUDIO_FILE)
    sample_rate = 16000  # Fixed target sample rate

    # Run Whisper ASR for transcription with word timestamps
    transcription_result = transcribe_audio(whisper_model, MEETING_AUDIO_FILE)
    word_list = [word for segment in transcription_result.get("segments", []) for word in segment.get("words", [])]

    # Run speaker diarization
    diarization = diarize_audio(diarization_pipeline, str(MEETING_AUDIO_FILE))

    speaker_map = {}      # Map from diarizer speaker label to enrolled speaker ID or Unknown label
    speaker_turns = []    # Collect speaker-labeled time segments

    # Iterate diarization segments and identify speakers
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
            identified_speaker, score = identify_speaker(segment_embedding, index,SIMILARITY_THRESHOLD)

            # Assign unknown speaker if confidence low
            if not identified_speaker:
                # Generate unique name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:6]
                identified_speaker = f"Spk_{timestamp}_{unique_id}"
                
                # Save audio segment (non-blocking failure)
                try:
                    audio_filename = UNKNOWN_VOICES_DIR / f"{identified_speaker}.wav"
                    torchaudio.save(str(audio_filename), segment_waveform.cpu(), sample_rate)
                except Exception as e:
                    logging.error(f"❌ Failed to save audio for {identified_speaker}: {e}")
                
                # Upsert embedding to Pinecone
                vector = segment_embedding.cpu().numpy().tolist()
                upsert_embeddings(index, [{'id': identified_speaker, 'values': vector}])
                
                logging.info(f"⬆️✨Enrolled new speaker: {identified_speaker}")

            speaker_map[speaker_label] = identified_speaker

        speaker_turns.append({'start': segment.start, 'end': segment.end, 'speaker': identified_speaker})

    # Extract words aligned to speaker turns within margin, build transcript lines
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

    # Write final transcript to output file and print
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for speaker, text in merged_lines:
            output_line = f"{speaker}: {text}"
            print(output_line)
            f.write(output_line + "\n")

    logging.info(f"Transcript saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
