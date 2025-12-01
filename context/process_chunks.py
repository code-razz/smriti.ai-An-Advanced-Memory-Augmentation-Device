"""
Helper module for chunking conversation text and storing in vector database.
Can be imported and used by server.py for real-time processing.
"""
import os
import sys
import uuid
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
import cohere
from pinecone import Pinecone, ServerlessSpec
import re
from db.db_utils2 import save_chunk_to_mongodb

# Load environment variables
load_dotenv()

# Log file for Pinecone upserts
LOG_FILE = Path(__file__).parent / "pinecone_upsert_log.txt"

# Load credentials
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_REGION = os.getenv("PINECONE_REGION")
NAMESPACE = "conversations"

# Initialize Cohere and Pinecone
co = None
pc = None
index = None

def get_cohere_client():
    """Get or initialize Cohere client."""
    global co
    if co is None:
        co = cohere.Client(COHERE_API_KEY)
    return co

def get_pinecone_index():
    """Get or initialize Pinecone index."""
    global pc, index
    if pc is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if index is None:
        # Create index if it doesn't exist
        if not pc.has_index(PINECONE_INDEX):
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
            )
        index = pc.Index(PINECONE_INDEX)
    
    return index

def split_into_utterances(conversation: str) -> List[tuple]:
    """
    Parse conversation into a list of (speaker, text) tuples.
    Expects lines with 'Speaker: text'. If no speaker label is found, speaker='Unknown'.
    """
    utterances = []
    lines = [ln.strip() for ln in conversation.splitlines() if ln.strip()]
    if not lines:
        return []
    speaker_re = re.compile(r'^\s*([^:]{1,50}):\s*(.*)$')
    for ln in lines:
        m = speaker_re.match(ln)
        if m:
            speaker = m.group(1).strip()
            text = m.group(2).strip()
        else:
            speaker = "Unknown"
            text = ln
        utterances.append((speaker, text))
    return utterances

def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences, keeping punctuation.
    Uses a conservative regex: split on (?<=[.!?])\s+.
    """
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return []
    return parts

def chunk_conversation_text(conversation_text: str, max_chars: int = 900, 
                           tz_name: str = "Asia/Kolkata") -> List[Dict]:
    """
    Chunk conversation text into structured chunks.
    
    Parameters:
        conversation_text: The diarized conversation text (format: "Speaker: text")
        max_chars: Maximum characters per chunk
        tz_name: Timezone name
    
    Returns:
        List of chunk dictionaries with text and metadata
    """
    if not conversation_text or not conversation_text.strip():
        return []
    
    utterances = split_into_utterances(conversation_text)
    if not utterances:
        return []
    
    # Base conversation ID same for all chunks
    base_conv_id = f"conv_{datetime.now(ZoneInfo(tz_name)).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    conversation_chunks = []
    buffer_lines: List[str] = []
    buffer_len = 0
    
    def flush_buffer():
        nonlocal buffer_lines, buffer_len
        if not buffer_lines:
            return
        text = "\n".join(buffer_lines)
        participants = []
        for bl in buffer_lines:
            sp = bl.split(":", 1)[0].strip()
            if sp not in participants:
                participants.append(sp)
        ts = datetime.now(ZoneInfo(tz_name)).isoformat()
        conversation_chunks.append({
            "text": text,
            "metadata": {
                "conversation_id": base_conv_id,
                "timestamp": ts,
                "location": "not_provided",
                "participants": participants,
                "tags": "not_provided"
            }
        })
        buffer_lines = []
        buffer_len = 0
    
    # Process each utterance and sentences
    for speaker, text in utterances:
        sentences = split_sentences(text)
        if not sentences:
            continue
        
        for sent in sentences:
            # Determine length increase
            if buffer_lines and buffer_lines[-1].startswith(f"{speaker}:"):
                length_increase = 1 + len(sent)  # space + sentence
            else:
                length_increase = len(speaker) + 2 + len(sent)  # speaker + ": " + sentence
            
            # Flush if needed
            if buffer_lines and (buffer_len + length_increase > max_chars):
                flush_buffer()
            
            # Add sentence to buffer
            if buffer_lines and buffer_lines[-1].startswith(f"{speaker}:"):
                buffer_lines[-1] = buffer_lines[-1] + " " + sent
                buffer_len += 1 + len(sent)
            else:
                new_line = f"{speaker}: {sent}"
                buffer_lines.append(new_line)
                buffer_len += len(new_line) + 1
    
    # Flush remaining
    flush_buffer()
    
    return conversation_chunks

def log_upsert_to_file(vectors: List[Dict], timestamp: str):
    """
    Log upserted vectors to a text file with timestamp marker.
    
    Parameters:
        vectors: List of vector dictionaries that were upserted
        timestamp: Timestamp string for the marker
    """
    try:
        # Ensure log file exists or create it
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare log entry
        log_entry = {
            "timestamp": timestamp,
            "count": len(vectors),
            "vectors": []
        }
        
        # Extract relevant information from each vector (without embedding values)
        for vec in vectors:
            log_entry["vectors"].append({
                "id": vec.get("id", "unknown"),
                "metadata": vec.get("metadata", {})
            })
        
        # Append to log file
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            # Write timestamp marker
            f.write(f"\n{'='*80}\n")
            f.write(f"TIMESTAMP: {timestamp}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total vectors upserted: {len(vectors)}\n")
            f.write(f"\n")
            
            # Write each vector record
            for i, vec_info in enumerate(log_entry["vectors"], 1):
                f.write(f"--- Vector {i} ---\n")
                f.write(f"ID: {vec_info['id']}\n")
                f.write(f"Metadata: {json.dumps(vec_info['metadata'], indent=2, ensure_ascii=False)}\n")
                f.write(f"\n")
            
            f.write(f"{'='*80}\n\n")
        
        print(f"📝 Logged {len(vectors)} upserted records to {LOG_FILE}")
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to log upsert to file: {e}")


def store_chunks_in_vector_db(conversation_chunks: List[Dict]) -> bool:
    """
    Store conversation chunks in Pinecone vector database.
    Logs all upserted records to a text file with timestamp markers.
    
    Parameters:
        conversation_chunks: List of chunk dictionaries
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not conversation_chunks:
        print("⚠️ No conversation chunks to store")
        return False
    
    try:
        co = get_cohere_client()
        index = get_pinecone_index()
        
        print(f"📝 Processing {len(conversation_chunks)} conversation chunks...")
        texts = [chunk["text"] for chunk in conversation_chunks]
        
        # Generate embeddings
        print("🔢 Generating embeddings...")
        embeddings = co.embed(
            texts=texts,
            input_type="search_document",
            model="embed-english-v3.0"
        ).embeddings
        
        print(f"🔢 Generated {len(embeddings)} embeddings")
        
        # Build vectors with metadata
        print("🔨 Building vectors with metadata...")
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(conversation_chunks, embeddings)):
            chunk_id = f"{chunk['metadata'].get('conversation_id', 'unknown')}_chunk_{i+1}"
            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    **chunk["metadata"],
                    "text": chunk["text"],
                    "chunk_index": i + 1
                }
            })
        
        print(f"📦 Built {len(vectors)} vectors ready for upload")
        
        # Upsert to Pinecone
        print("🚀 Uploading vectors to Pinecone...")
        index.upsert(
            vectors=vectors,
            namespace=NAMESPACE
        )
        
        print(f"✅ Successfully embedded and upserted {len(vectors)} chunks into Pinecone (namespace: '{NAMESPACE}')")
        
        # Save to MongoDB
        print("💾 Saving chunks to MongoDB...")
        for vec in vectors:
            # Reconstruct chunk object for MongoDB
            # Exact duplicate of Pinecone structure: id, values, metadata
            mongo_chunk = {
                "id": vec["id"],
                "values": vec["values"],
                "metadata": vec["metadata"]
            }
            save_chunk_to_mongodb(mongo_chunk)
        print(f"✅ Saved {len(vectors)} chunks to MongoDB")
        
        # Log upserted records to file with timestamp
        timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S %Z")
        log_upsert_to_file(vectors, timestamp)
        
        return True
        
    except Exception as e:
        print(f"❌ Error storing chunks in vector DB: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_and_store_conversation(conversation_text: str) -> bool:
    """
    Complete pipeline: chunk conversation text and store in vector database.
    
    Parameters:
        conversation_text: The diarized conversation text
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("📦 Chunking conversation...")
    chunks = chunk_conversation_text(conversation_text)
    
    if not chunks:
        print("⚠️ No chunks generated from conversation")
        return False
    
    print(f"✅ Generated {len(chunks)} chunks")
    
    return store_chunks_in_vector_db(chunks)