from mongodb_client import chunks_collection, alerts_collection

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

def save_alert_to_mongodb(alert: dict):
    """
    Creates or updates an alert in MongoDB.
    """
    if alerts_collection is None:
        print("[MongoDB] Skipping alert save: alerts_collection is not initialized.")
        return

    try:
        # Ensure the alert has an 'id' field
        if "id" not in alert:
            # Generate one if missing, or use a specific logic
            import uuid
            alert["id"] = str(uuid.uuid4())

        alerts_collection.update_one(
            {"id": alert["id"]},          # use alert ID as filter
            {"$set": alert},              # update or insert
            upsert=True
        )
        print(f"[🔔MongoDB] Saved alert: {alert['id']}") 
    except Exception as e:
        print(f"[MongoDB] Error saving alert {alert.get('id', 'unknown')}: {e}")
