from db_utils import save_chunk_to_mongodb

# A sample conversation chunk like the one in your Pinecone UI screenshot
sample_chunk = {
    "id": "conv_20250922_023036_cf6578_chunk_1",      # this becomes MongoDB _id
    "text": (
        "User: Did you bring the medicine?\n"
        "Alex: Yes, I picked it up on the way.\n"
        "User: Great, I need to take it after dinner.\n"
        "Alex: Remember, the doctor said to avoid coffee."
    ),
    "metadata": {
        "chunk_index": 1,
        "conversation_id": "conv_20250922_023036_cf6578",
        "location": "not_provided",
        "participants": ["User", "Alex"],
        "tags": "not_provided",
        "timestamp": "2025-09-22T02:30:36.120356+05:30"
    }
}

if __name__ == "__main__":
    print("Saving sample chunk into MongoDB...")
    save_chunk_to_mongodb(sample_chunk)
    print("Done! Check MongoDB Atlas → smriti-ai → conversation_chunks")
