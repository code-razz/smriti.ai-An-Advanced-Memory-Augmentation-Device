# pip install "pinecone[grpc]"
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

# PINECONE_INDEX = os.getenv("PINECONE_INDEX")
# COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
# PINECONE_REGION = os.getenv("PINECONE_REGION")
NAMESPACE = "conversations"  # You can customize or parametrize this


pc = Pinecone(api_key=PINECONE_API_KEY)

# To get the unique host for an index, 
# see https://docs.pinecone.io/guides/manage-data/target-an-index
index = pc.Index(host=PINECONE_INDEX_HOST)

result=index.fetch(ids=['chunk-1', 'chunk-10', 'chunk-11', 'chunk-12'], namespace=NAMESPACE)
# print(result)

for ids in result.vectors:
    print(ids)
    # print(result.vectors[ids].values)
    print(result.vectors[ids].metadata)
    print("-----")