from mongodb_client import chunks_collection

def save_chunk_to_mongodb(chunk: dict):
    """
    Creates or updates a conversation chunk in MongoDB.
    """
    if chunks_collection is None:
        print("[MongoDB] Skipping upsert: chunks_collection is not initialized.")
        return

    try:
        # Ensure the chunk has an 'id' field which matches the Pinecone ID
        if "id" not in chunk:
            print("[MongoDB] Error: Chunk missing 'id' field.")
            return

        chunks_collection.update_one(
            {"_id": chunk["id"]},         # use Pinecone ID as MongoDB _id
            {"$set": chunk},              # update or insert
            upsert=True
        )
        print(f"[💾MongoDB] Saved chunk: {chunk['id']}") # Commented out to reduce noise, or keep for debug
    except Exception as e:
        print(f"[MongoDB] Error saving chunk {chunk.get('id', 'unknown')}: {e}")
