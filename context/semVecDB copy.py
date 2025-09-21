import os
import uuid
from dotenv import load_dotenv
import cohere
from pinecone import Pinecone, ServerlessSpec
# Import conversation chunks from the chunker module
from chunker_from_stt_diarization import conversation_chunks

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

# === Step 1: Validate and prepare documents ===
if not conversation_chunks:
    print("❌ No conversation chunks found. Please run chunker_from_stt_diarization.py first.")
    exit(1)

print(f"📝 Processing {len(conversation_chunks)} conversation chunks...")
texts = [chunk["text"] for chunk in conversation_chunks]
# ids = [f"chunk-{i+1}" for i in range(len(conversation_chunks))]

embeddings = co.embed(
    texts=texts,  # ✅ keyword argument
    input_type="search_document",
    model="embed-english-v3.0"
).embeddings

print(f"🔢 Generated {len(embeddings)} embeddings")


# === Step 2: Build vectors with metadata ===
print("🔨 Building vectors with metadata...")
vectors = []
for i, (chunk, embedding) in enumerate(zip(conversation_chunks, embeddings)):
    # Create unique ID for each chunk to avoid overwriting
    chunk_id = f"{chunk['metadata'].get('conversation_id', 'unknown')}_chunk_{i+1}"
    vectors.append({
        "id": chunk_id,
        "values": embedding,
        "metadata": {
            **chunk["metadata"],
            "text": chunk["text"],
            "chunk_index": i + 1  # Add chunk index for reference
        }
    })

print(f"📦 Built {len(vectors)} vectors ready for upload")
print("🔍 Sample vector IDs:")
for i, vector in enumerate(vectors[:3]):  # Show first 3 IDs
    print(f"  - {vector['id']}")
if len(vectors) > 3:
    print(f"  ... and {len(vectors) - 3} more")

# === Step 3: Upsert to Pinecone ===
print("🚀 Uploading vectors to Pinecone...")
index.upsert(
    vectors=vectors,
    namespace=NAMESPACE
)

print(f"✅ Successfully embedded and upserted {len(vectors)} chunks into Pinecone (namespace: '{NAMESPACE}')")

# Example run if module executed
if __name__ == "__main__":
    print("🎉 Semantic vector database setup complete!")
