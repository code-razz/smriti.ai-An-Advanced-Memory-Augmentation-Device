from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")

# It's better not to raise an error immediately on import if env var is missing, 
# but rather when trying to connect, or handle it gracefully.
# However, following the user's snippet pattern but making it robust.

client = None
db = None
chunks_collection = None

if MONGO_URL:
    try:
        client = MongoClient(MONGO_URL)
        db = client["smriti-ai"]  # database name
        chunks_collection = db["conversation_chunks"]  # collection
        print("[MongoDB] Client initialized.")
    except Exception as e:
        print(f"[MongoDB] Error initializing client: {e}")
else:
    print("[MongoDB] Warning: MONGO_URL not found in environment variables.")
