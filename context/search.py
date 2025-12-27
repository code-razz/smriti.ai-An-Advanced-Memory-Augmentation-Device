# query.py

import os
from dotenv import load_dotenv
import cohere
from pinecone import Pinecone
import time

load_dotenv()

# Load API Keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Initialize
co = cohere.Client(COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def get_query_and_memories(query_text=None):
    # Search query
    if query_text:
        query = query_text
    else:
        query = input("Ask AISmriti your question: ")
    # query = "What did I eat with my friend?"
    # query = "What English Vocabulary is good?"
    # query = "What did the user say about the medicine and doctor?"
    # query = "What did I eat with my friend?"
    
    # query = "What English Vocabulary is good?"
    # query = "What did the user say about the medicine and doctor?"
    # query = "Whose point was that people will be lazy?"

    # Embed query
    print(f"🕒 [PERF] Embedding (Cohere) Started at {time.strftime('%H:%M:%S')}")
    emb_start = time.time()
    query_embedding = co.embed(
        texts=[query],
        input_type="search_query",
        model="embed-english-v3.0"
    ).embeddings[0]
    print(f"🕒 [PERF] Embedding (Cohere) Completed at {time.strftime('%H:%M:%S')}")
    print(f"🕒 [PERF] Embedding (Cohere) Completed in {time.time() - emb_start:.4f}s")

    # Query top 3
    print(f"🕒 [PERF] Vector Search (Pinecone) Started at {time.strftime('%H:%M:%S')}")
    pf_start = time.time()
    results = index.query(
        namespace="conversations",
        vector=query_embedding,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    print(f"🕒 [PERF] Vector Search (Pinecone) Completed at {time.strftime('%H:%M:%S')}")
    print(f"🕒 [PERF] Vector Search (Pinecone) Completed in {time.time() - pf_start:.4f}s")

    # Display results
    print("\n=====================================================================")
    print("\n🔍 Top Matches:")
    memories = []
    for match in results["matches"]:
        score = match["score"]
        meta = match["metadata"]
        memory_entry = {
            "score": score,
            "text": meta.get("text"),
            "location": meta.get("location"),
            "participants": meta.get("participants"),
            "timestamp": meta.get("timestamp"),
            "conversation_id": meta.get("conversation_id"),
        }
        memories.append(memory_entry)
        print(f"\nScore: {match['score']:.4f}")
        print(f"Text: {match['metadata']['text']}")
        print(f"Location: {match['metadata'].get('location')}")
        print(f"Participants: {match['metadata'].get('participants')}")
        print(f"Timestamp: {match['metadata'].get('timestamp')}")
        print(f"Conversation ID: {match['metadata'].get('conversation_id')}")

    print("\n=====================================================================")
        
    return query, memories
