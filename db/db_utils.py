from db.mongodb_client import chunks_collection
import sys
from pathlib import Path

# Add context directory to path to import process_chunks
# sys.path.append(str(Path(_file_).parent / "context"))

from context.process_chunks import get_cohere_client, get_pinecone_index, NAMESPACE

def update_speaker_name_in_mongodb(old_name: str, new_name: str):
    """
    Updates:
    - metadata.participants
    - metadata.text
    
    And syncs changes to Pinecone.
    """

    # Find all documents that contain speaker name
    docs = list(chunks_collection.find({
        "$or": [
            {"metadata.participants": old_name},
            {"metadata.text": {"$regex": old_name}}
        ]
    }))

    print(f"[MongoDB] Found {len(docs)} docs to update")

    updated_count = 0
    pinecone_updated_count = 0
    
    # Initialize clients
    try:
        co = get_cohere_client()
        index = get_pinecone_index()
        pinecone_available = True
    except Exception as e:
        print(f"⚠️ Pinecone/Cohere initialization failed: {e}")
        pinecone_available = False

    for doc in docs:
        # Update participants
        participants = doc["metadata"].get("participants", [])
        participants = [
            new_name if p == old_name else p
            for p in participants
        ]

        # Update text inside metadata
        old_text = doc["metadata"].get("text", "")
        new_text = old_text.replace(old_name, new_name)

        # Save updated data to MongoDB
        chunks_collection.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "metadata.participants": participants,
                    "metadata.text": new_text
                }
            }
        )

        updated_count += 1
        
        # Sync to Pinecone
        if pinecone_available:
            try:
                # 1. Get Pinecone ID (stored as 'id' in MongoDB doc usually, or we construct it)
                pinecone_id = doc.get("id")
                if not pinecone_id:
                    print(f"⚠️ Could not find 'id' in MongoDB doc {doc['_id']}, skipping Pinecone sync.")
                    continue
                
                # 2. Delete old vector
                index.delete(ids=[pinecone_id], namespace=NAMESPACE)
                
                # 3. Generate new embedding
                # We need to re-embed because the text changed (speaker name in text)
                response = co.embed(
                    texts=[new_text],
                    input_type="search_document",
                    model="embed-english-v3.0"
                )
                new_embedding = response.embeddings[0]
                
                # 4. Upsert new vector
                new_metadata = doc.get("metadata", {}).copy()
                new_metadata["participants"] = participants
                new_metadata["text"] = new_text
                
                vector = {
                    "id": pinecone_id,
                    "values": new_embedding,
                    "metadata": new_metadata
                }
                
                index.upsert(vectors=[vector], namespace=NAMESPACE)
                pinecone_updated_count += 1
                print(f"✅ Synced doc {pinecone_id} to Pinecone")
                
            except Exception as e:
                print(f"❌ Failed to sync doc {doc.get('id', 'unknown')} to Pinecone: {e}")

    print(f"[MongoDB] Updated {updated_count} documents")
    if pinecone_available:
        print(f"[Pinecone] Updated {pinecone_updated_count} vectors")

    return {"updated_docs": updated_count, "pinecone_updated": pinecone_updated_count}

update_speaker_name_in_mongodb("ATIA","Barsha")
# 1: "manya"