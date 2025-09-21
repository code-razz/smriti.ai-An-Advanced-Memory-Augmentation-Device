from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, INDEX_NAME, EMBEDDING_DIM

def get_pinecone_index():
    """
    Initialize Pinecone client and ensure the index exists.

    Returns:
        index (pinecone.Index): Pinecone index instance.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if it doesn't exist
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(INDEX_NAME)

# def vector_already_enrolled(index, speaker_id):
#     """
#     Check if a speaker's embedding vector is already enrolled in Pinecone.

#     Parameters:
#         index: Pinecone index instance.
#         speaker_id (str): Unique speaker ID.

#     Returns:
#         bool: True if speaker embedding exists, False otherwise.
#     """
#     response = index.fetch(ids=[speaker_id], namespace="reference")
#     return speaker_id in response.vectors


def upsert_embeddings(index, vectors):
    """
    Upload new speaker embedding vectors to Pinecone.

    Parameters:
        index (pinecone.Index): Pinecone index instance.
        vectors (list): List of dictionaries with 'id', 'values', and optionally 'metadata'.
    """
    index.upsert(vectors=vectors, namespace="reference")

def find_matching_speaker(index, embedding_tensor, threshold=0.6, top_k=1):
    """
    Query Pinecone with an embedding vector to find closest enrolled speaker
    (by similarity, not by ID). Returns (speaker_id, similarity_score), or (None, None).
    """
    vector = embedding_tensor.cpu().numpy().tolist()
    response = index.query(
        vector=vector,
        namespace="reference",
        top_k=top_k,
        include_values=False,
        include_metadata=True
    )
    if not response.matches:
        return None, None
    best_match = response.matches[0]
    if best_match.score >= threshold:
        return best_match.id, best_match.score
    else:
        return None, best_match.score
