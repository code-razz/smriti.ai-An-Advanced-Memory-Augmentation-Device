import os
import uuid
from dotenv import load_dotenv
import cohere
from pinecone import Pinecone, ServerlessSpec
from conversation_chunks2 import conversation_chunks

# Load credentials from .env
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_REGION = os.getenv("PINECONE_REGION")
NAMESPACE = "conversations"  # You can customize or parametrize this

# Initialize Cohere and Pinecone
co = cohere.Client(COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index if it doesn't exist
if not pc.has_index(PINECONE_INDEX):
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )

index = pc.Index(PINECONE_INDEX)

# === Step 1: Embed documents ===
texts = [chunk["text"] for chunk in conversation_chunks]
# ids = [f"chunk-{i+1}" for i in range(len(conversation_chunks))]

embeddings = co.embed(
    texts=texts,  # ✅ keyword argument
    input_type="search_document",
    model="embed-english-v3.0"
).embeddings


# === Step 2: Build vectors with metadata ===
vectors = []
for i, (chunk, embedding) in enumerate(zip(conversation_chunks, embeddings)):
    vectors.append({
        "id": chunk["metadata"].get("conversation_id"),  # Use conversation_id or fallback to chunk-{i+1}
        "values": embedding,
        "metadata": {
            **chunk["metadata"],
            "text": chunk["text"]
        }
    })

# === Step 3: Upsert to Pinecone ===
index.upsert(
    vectors=vectors,
    namespace=NAMESPACE
)

print(f"✅ Successfully embedded and upserted {len(vectors)} chunks into Pinecone (namespace: '{NAMESPACE}')")
