import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST1")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "reference")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)


def update_speaker_name_in_pinecone(old_name: str, new_name: str):
    print(f"🔍 Fetching Pinecone vector '{old_name}'...")
    fetch_result = index.fetch(ids=[old_name], namespace=NAMESPACE)

    if old_name not in fetch_result.vectors:
        raise RuntimeError(f"❌ ID '{old_name}' not found in Pinecone")

    vec = fetch_result.vectors[old_name]

    values = vec.values

    # Extract ONLY audio_url if present
    metadata = vec.metadata or {}
    audio_url = metadata.get("audio_url")

    safe_metadata = {}
    if audio_url:
        safe_metadata["audio_url"] = audio_url

    print(f"📦 Creating new vector '{new_name}' (keeping audio_url={audio_url})...")

    index.upsert(
        vectors=[
            {
                "id": new_name,
                "values": values,
                "metadata": safe_metadata   # ONLY audio_url
            }
        ],
        namespace=NAMESPACE
    )

    # Verify new vector exists before deleting old one
    check = index.fetch(ids=[new_name], namespace=NAMESPACE)
    if new_name not in check.vectors:
        raise RuntimeError(f"❌ Could not verify new vector '{new_name}'")

    print(f"🗑️ Deleting old Pinecone entry '{old_name}'...")
    index.delete(ids=[old_name], namespace=NAMESPACE)

    print(f"✅ Renamed speaker in Pinecone: {old_name} → {new_name}")
    return {"pinecone_updated": True, "audio_url_kept": audio_url}
