from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")

if not MONGO_URL:
    raise RuntimeError("MONGO_URL is missing in .env file")

client = MongoClient(MONGO_URL)

db = client["smriti-ai"]  # database name
chunks_collection = db["conversation_chunks"]  # collection
