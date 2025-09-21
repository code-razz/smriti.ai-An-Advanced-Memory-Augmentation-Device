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

# Implicit pagination using a generator function
for ids in index.list(prefix="chunk-1", namespace=NAMESPACE):
    print(ids)

# # Manual pagination
# results = index.list_paginated(
#     prefix="chunk-1",
#     limit=3,
#     namespace=NAMESPACE,
#     pagination_token="eyJza2lwX3Bhc3QiOiIxMDEwMy0="
# )

# print(results)

'''
pagination_token = None

while True:
    results = index.list_paginated(
        prefix="doc1#",
        limit=3,
        namespace="example-namespace",
        pagination_token=pagination_token
    )

    for id in results.vectors:
        print(id)

    if results.pagination_token:
        pagination_token = results.pagination_token
    else:
        break  # No more results
'''