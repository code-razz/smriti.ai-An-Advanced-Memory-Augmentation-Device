import os
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from pydantic import BaseModel
from db.db_utils import update_speaker_name_in_mongodb
from db.pinecone_update import update_speaker_name_in_pinecone


load_dotenv()

# Simple config - just set these in .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST1")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "reference")  # Default namespace

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set in .env file")

if not PINECONE_INDEX_HOST:
    raise RuntimeError("PINECONE_INDEX_HOST1 is not set in .env file")

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)


FACE_INDEX_HOST=os.getenv("FACE_INDEX_HOST")

if not FACE_INDEX_HOST:
    raise RuntimeError("PINECONE_INDEX_HOST1 is not set in .env file")

index2 = pc.Index(host=FACE_INDEX_HOST)


class RenameRequest(BaseModel):
    old_name: str
    new_name: str

app = FastAPI(title="Smriti Voice API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/rename-speaker")
def rename_speaker(payload: dict):

    old_name = payload.get("old_name")
    new_name = payload.get("new_name")

    if not old_name or not new_name:
        return {"error": "old_name and new_name are required"}

    print(f"\n🔄 Rename request received: {old_name} → {new_name}")

    # Update MongoDB
    mongo_result = update_speaker_name_in_mongodb(old_name, new_name)

    # Update Pinecone
    pinecone_result = update_speaker_name_in_pinecone(old_name, new_name)

    return {
        "message": "Rename successful",
        "mongo": mongo_result,
        "pinecone": pinecone_result
    }
    

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/speakers")
def get_speakers(limit: int = 100):
    """
    Retrieve ALL speaker IDs dynamically from Pinecone using a dummy-vector query.
    No hardcoding, no separate database.
    """
    try:
        # 1️⃣ Describe index stats to know the vector dimension
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        ns_info = namespaces.get(NAMESPACE, {})

        vector_count = ns_info.get("vector_count", 0)
        dimension = stats.get("dimension", None)

        if vector_count == 0:
            return {"items": [], "count": 0}

        if not dimension:
            raise RuntimeError("Failed to determine index vector dimension.")

        # 2️⃣ Create dummy zero-vector of correct dimension
        dummy_vector = [0.0] * dimension

        # 3️⃣ Query Pinecone to retrieve ALL vectors
        # Increase top_k when your data grows
        top_k = min(vector_count, 10000)  # safe max limit

        results = index.query(
            namespace=NAMESPACE,
            vector=dummy_vector,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )

        # 4️⃣ Extract IDs & metadata
        matches = results.get("matches", [])

        items = []
        for match in matches:
            items.append({
                "id": match.get("id"),
                "name": match.get("id"),
                "source": match.get("metadata", {}).get("source", None),
                "audio_url": match.get("metadata", {}).get("audio_url")
            })

        # 5️⃣ Return limit
        return {
            "items": items,
            "count": len(items)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/update-speaker")
def update_speaker(payload: dict):

    old_name = payload.get("old_name")
    new_name = payload.get("new_name")

    if not old_name or not new_name:
        return {"error": "old_name and new_name are required"}

    print(f"\n🔄 Rename request received: {old_name} → {new_name}")

    # 1️⃣ Update MongoDB
    mongo_result = update_speaker_name_in_mongodb(old_name, new_name)

    # 2️⃣ Update Pinecone (MANDATORY)
    pinecone_result = update_speaker_name_in_pinecone(old_name, new_name)

    return {
        "message": "Rename completed.",
        "mongo": mongo_result,
        "pinecone": pinecone_result
    }

@app.get("/dashboard-stats")
def dashboard_stats():
    # ---- Speaker Stats ----
    stats_speakers = index.describe_index_stats()
    ns_speakers = stats_speakers.get("namespaces", {}).get(NAMESPACE, {})
    total_speakers = ns_speakers.get("vector_count", 0)

    # Count unknown speakers
    dimension = stats_speakers.get("dimension", 0)
    dummy = [0.0] * dimension
    results = index.query(
        vector=dummy,
        top_k=total_speakers,
        namespace=NAMESPACE,
        include_metadata=False
    )
    matches = results.matches if hasattr(results, "matches") else results.get("matches", [])
    unknown_count = sum(1 for m in matches if m.id.startswith("Spk_"))

    # ---- Face Stats ----
    stats_faces = index2.describe_index_stats()
    ns_faces = stats_faces.get("namespaces", {}).get("faces", {})
    total_faces = ns_faces.get("vector_count", 0)

    return {
        "total_speakers": total_speakers,
        "unknown_speakers": unknown_count,
        "total_faces": total_faces
    }


@app.get("/debug-face-stats")
def debug_face_stats():
    index = pc.Index(host=FACE_INDEX_HOST)
    return index.describe_index_stats()

@app.post("/rename-face")
def rename_face(payload: dict):
    face_id = payload.get("id")
    old_name = payload.get("old_name")
    new_name = payload.get("new_name")

    if not face_id or not new_name:
        raise HTTPException(status_code=400, detail="id and new_name are required")

    try:
        # MUST include namespace="faces"
        index2.update(
            id=face_id,
            namespace="faces",
            set_metadata={"name": new_name}
        )

        return {
            "message": "Face renamed successfully",
            "id": face_id,
            "old_name": old_name,
            "new_name": new_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/faces")
def get_faces(limit: int = 100):
    """
    Fetch all face embeddings from the Pinecone 'faces' namespace.
    """

    try:
        stats = index2.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        ns_info = namespaces.get("faces", {}) or namespaces.get("reference", {})

        vector_count = ns_info.get("vector_count", 0)
        dimension = stats.get("dimension", None)

        if vector_count == 0:
            return {"items": [], "count": 0}

        if not dimension:
            raise RuntimeError("Could not detect vector dimension")

        dummy_vector = [0.0] * dimension

        results = index2.query(
            namespace="faces",
            vector=dummy_vector,
            top_k=min(vector_count, 10000),
            include_metadata=True,
            include_values=False
        )

        items = []
        for m in results.matches:
            md = m.metadata or {}
            items.append({
                "id": m.id,
                "name": md.get("name", m.id),
                "image_url": md.get("image_url"),
                "created_at": md.get("created_at")
            })

        return {"items": items, "count": len(items)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
