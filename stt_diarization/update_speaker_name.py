import sys
import os
from pathlib import Path

# Ensure we can import modules from the current directory
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from pinecone_utils import get_pinecone_index

def update_speaker_name(old_name, new_name):
    """
    Updates a speaker's name in Pinecone by creating a new vector with the new name
    and deleting the old one.
    """
    print(f"Initializing Pinecone index...")
    try:
        index = get_pinecone_index()
    except Exception as e:
        print(f"Error initializing Pinecone index: {e}")
        return False
    
    print(f"Fetching vector for '{old_name}'...")
    # Fetch the vector by ID
    try:
        fetch_response = index.fetch(ids=[old_name], namespace="reference")
    except Exception as e:
        print(f"Error fetching vector: {e}")
        return False
    
    if old_name not in fetch_response.vectors:
        print(f"Error: Speaker '{old_name}' not found in Pinecone.")
        return False
    
    vector_data = fetch_response.vectors[old_name]
    embedding_values = vector_data.values
    metadata = vector_data.metadata if vector_data.metadata else {}
    
    print(f"Found speaker '{old_name}'. Creating new entry for '{new_name}'...")
    
    # Upsert with new ID
    # We keep the same embedding values and metadata
    new_vector = {
        'id': new_name,
        'values': embedding_values,
        'metadata': metadata
    }
    
    try:
        index.upsert(vectors=[new_vector], namespace="reference")
        print(f"Successfully created '{new_name}'.")
        
        # Verify creation before deletion (paranoid check)
        check_response = index.fetch(ids=[new_name], namespace="reference")
        if new_name in check_response.vectors:
            print(f"Deleting old entry '{old_name}'...")
            index.delete(ids=[old_name], namespace="reference")
            print(f"Successfully renamed '{old_name}' to '{new_name}'.")
            return True
        else:
            print(f"Error: Failed to verify creation of '{new_name}'. Aborting deletion of '{old_name}'.")
            return False
            
    except Exception as e:
        print(f"An error occurred during upsert/delete: {e}")
        return False

if __name__ == "__main__":
    print("--- Update Speaker Name in Pinecone ---")
    
    if len(sys.argv) == 3:
        old_name = sys.argv[1]
        new_name = sys.argv[2]
    else:
        old_name = "Spk_20251125_235946_895934"
        # old_name = input("Enter the current speaker ID (name) to change: ").strip()
        if not old_name:
            print("Old name cannot be empty.")
            sys.exit(1)
            
        new_name = "Test_Name"
        # new_name = input("Enter the new speaker name: ").strip()
        if not new_name:
            print("New name cannot be empty.")
            sys.exit(1)
            
    update_speaker_name(old_name, new_name)
