# query.py

import os
from dotenv import load_dotenv
import cohere
from pinecone import Pinecone

load_dotenv()

# Load API Keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Initialize
co = cohere.Client(COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Search query
# query = "What did I eat with my friend?"
query = "What English Vocabulary is good?"
# query = "What did the user say about the medicine and doctor?"

# Embed query (✅ FIXED)
query_embedding = co.embed(
    texts=[query],
    input_type="search_query",
    model="embed-english-v3.0"
).embeddings[0]

# Query top 3
results = index.query(
    namespace="conversations",
    vector=query_embedding,
    top_k=3,
    include_values=False,
    include_metadata=True
)

# Display results
print("\n🔍 Top Matches:")
for match in results["matches"]:
    print(f"\nScore: {match['score']:.4f}")
    print(f"Text: {match['metadata']['text']}")
    print(f"Location: {match['metadata'].get('location')}")
    print(f"Participants: {match['metadata'].get('participants')}")
    print(f"Timestamp: {match['metadata'].get('timestamp')}")
    print(f"Conversation ID: {match['metadata'].get('conversation_id')}")
