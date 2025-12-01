from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST1")
REFERENCE_NAMESPACE = "reference"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)

def rename_speaker_embedding(old_name: str, new_name: str):
    """
    Renames a speaker embedding:
    - Fetches vector by old_name
    - Deletes old_name
    - Inserts same vector under new_name
    """

    # Fetch old speaker vector
    result = index.fetch(
        ids=[old_name],
        namespace=REFERENCE_NAMESPACE
    )

    if old_name not in result["vectors"]:
        return {"status": "not_found"}

    vec = result["vectors"][old_name]
    values = vec["values"]
    metadata = vec.get("metadata", {})

    # Delete old record
    index.delete(ids=[old_name], namespace=REFERENCE_NAMESPACE)

    # Insert new record
    index.upsert(
        vectors=[{
            "id": new_name,
            "values": values,
            "metadata": metadata
        }],
        namespace=REFERENCE_NAMESPACE
    )

    return {"status": "updated"}
