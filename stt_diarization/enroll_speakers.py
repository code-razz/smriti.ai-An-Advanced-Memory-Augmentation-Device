from config import REFERENCE_VOICES_DIR, SPEECHBRAIN_MODEL, DEVICE
from pinecone_utils import get_pinecone_index, upsert_embeddings, find_matching_speaker
from utils import load_and_resample
from speechbrain.inference.speaker import SpeakerRecognition

def enroll_speakers():
    speaker_id_model = SpeakerRecognition.from_hparams(
        source=SPEECHBRAIN_MODEL,
        savedir=f"pretrained_models/{SPEECHBRAIN_MODEL.replace('/', '_')}",
        run_opts={"device": DEVICE}
    )
    index = get_pinecone_index()
    vectors_to_upload = []

    for ref_file in REFERENCE_VOICES_DIR.glob("*.wav"):
        speaker_name = ref_file.stem.capitalize()
        waveform = load_and_resample(ref_file)
        embedding = speaker_id_model.encode_batch(waveform).squeeze()

        # NEW: Check by embedding, not by ID
        existing_id, score = find_matching_speaker(index, embedding, threshold=0.7)
        if existing_id:
            print(f"Speaker similar to {speaker_name} already enrolled as {existing_id} (score {score:.2f}). Skipping.")
            continue

        vectors_to_upload.append({
            "id": speaker_name,
            "values": embedding.cpu().numpy().tolist(),
            "metadata": {"source": "reference"}
        })

    if vectors_to_upload:
        upsert_embeddings(index, vectors_to_upload)
        print(f"Uploaded {len(vectors_to_upload)} new speaker embeddings.")
    else:
        print("No new speakers to enroll.")

if __name__ == "__main__":
    enroll_speakers()
