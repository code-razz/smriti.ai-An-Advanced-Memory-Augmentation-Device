import os
import sys
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION")
# Use a dedicated index for faces (128 dimensions)
# If PINECONE_FACE_INDEX is not set, default to "smriti-faces"
PINECONE_FACE_INDEX = os.getenv("PINECONE_FACE_INDEX", "smriti-faces")
FACE_NAMESPACE = "faces"

pc = None
face_index = None

def get_face_index():
    """Get or initialize Pinecone index for faces (128-d)."""
    global pc, face_index
    
    if pc is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if face_index is None:
        # Check if index exists
        if not pc.has_index(PINECONE_FACE_INDEX):
            print(f"⚠️ Index '{PINECONE_FACE_INDEX}' not found. Creating it (128-d)...")
            try:
                pc.create_index(
                    name=PINECONE_FACE_INDEX,
                    dimension=128,  # Face recognition embeddings are 128-d
                    metric="cosine", # or euclidean
                    spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
                )
                print(f"✅ Created index '{PINECONE_FACE_INDEX}'")
                # Wait a bit for initialization
                time.sleep(5)
            except Exception as e:
                print(f"❌ Failed to create index '{PINECONE_FACE_INDEX}': {e}")
                return None
        
        face_index = pc.Index(PINECONE_FACE_INDEX)
    
    return face_index

def search_face(embedding, threshold=0.6):
    """
    Search for a matching face in Pinecone.
    
    Args:
        embedding (list): 128-d face embedding.
        threshold (float): Similarity threshold.
    
    Returns:
        dict: Best match metadata or None if no match found.
    """
    try:
        index = get_face_index()
        if not index:
            return None
        
        # Query Pinecone
        results = index.query(
            vector=embedding,
            top_k=1,
            include_metadata=True,
            namespace=FACE_NAMESPACE
        )
        
        if not results['matches']:
            return None
            
        match = results['matches'][0]
        score = match['score']
        
        print(f"🔍 Best match: {match['metadata'].get('name')} (Score: {score:.4f})")
        
        # Threshold check (Cosine: 1.0 is best)
        # 0.92 is stricter (approx 0.40 Euclidean) to reduce false positives
        if score >= 0.92: 
            return match['metadata']
        else:
            print(f"⚠️ Match rejected: Score {score:.4f} < 0.92")
            return None

    except Exception as e:
        print(f"❌ Error searching face in Pinecone: {e}")
        return None

def enroll_face(name, embedding, image_url):
    """
    Enroll a new face into Pinecone.
    
    Args:
        name (str): Name of the person.
        embedding (list): 128-d face embedding.
        image_url (str): URL of the face image (Cloudinary).
        
    Returns:
        bool: True if successful.
    """
    try:
        index = get_face_index()
        if not index:
            return False
        
        face_id = f"face_{uuid.uuid4().hex}"
        timestamp = int(time.time())
        
        metadata = {
            "name": name,
            "image_url": image_url,
            "created_at": timestamp,
            "type": "face"
        }
        
        vector = {
            "id": face_id,
            "values": embedding,
            "metadata": metadata
        }
        
        index.upsert(
            vectors=[vector],
            namespace=FACE_NAMESPACE
        )
        
        print(f"✅ Enrolled face for '{name}' in Pinecone (Index: {PINECONE_FACE_INDEX}, ID: {face_id})")
        return True

    except Exception as e:
        print(f"❌ Error enrolling face in Pinecone: {e}")
        return False
