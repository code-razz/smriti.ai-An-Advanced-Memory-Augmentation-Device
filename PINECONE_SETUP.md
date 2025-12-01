# Pinecone Setup Guide

## What is a Pinecone Namespace?

A **namespace** in Pinecone is like a folder inside an index. It lets you organize different types of data in the same index without mixing them up.

### Simple Analogy
- **Index** = A filing cabinet
- **Namespace** = A drawer inside that cabinet
- **Vectors** = Documents stored in that drawer

### Why Use Namespaces?

1. **Organization**: Keep different data types separate (e.g., speaker embeddings vs conversation chunks)
2. **Isolation**: Query/search only within a specific namespace
3. **Flexibility**: Same index can hold multiple projects or data types

### Example Structure

```
Index: "my-vector-db"
├── Namespace: "english-sentences"  ← Your semantic search data
│   ├── Vector ID: 182
│   ├── Vector ID: 75
│   └── ...
├── Namespace: "reference"          ← Speaker embeddings
│   ├── Vector ID: "Aatya"
│   ├── Vector ID: "Manya"
│   └── ...
└── Namespace: "conversations"      ← Conversation chunks
    ├── Vector ID: "chunk-1"
    └── ...
```

## Environment Configuration (.env)

Create or update your `.env` file in the project root with the following:

```bash
# Required: Your Pinecone API key (same for all indexes)
PINECONE_API_KEY=your-api-key-here

# ============================================
# Semantic Search Index (for english-sentences namespace)
# ============================================
# Option 1: Use index name (if you know it)
SEMANTIC_INDEX=your-semantic-index-name

# Option 2: Use index host URL (more reliable)
# Get this from Pinecone dashboard → Your Index → Host
SEMANTIC_INDEX_HOST=https://your-index-xxxxx.svc.environment.pinecone.io

# Default namespace for semantic search
SEMANTIC_NAMESPACE=english-sentences

# ============================================
# Speaker Embeddings Index (for reference namespace)
# ============================================
# Option 1: Use index name
SPEAKER_INDEX=speaker-embeddings

# Option 2: Use index host URL
SPEAKER_INDEX_HOST=https://speaker-index-xxxxx.svc.environment.pinecone.io

# Default namespace for speaker embeddings
SPEAKER_NAMESPACE=reference
```

## How to Find Your Index Information

1. **Go to Pinecone Dashboard**: https://app.pinecone.io
2. **Select your project**
3. **Click on your index**
4. **Copy the "Host" URL** (looks like `https://xxx-xxx.svc.environment.pinecone.io`)
5. **Check namespaces**: In the index view, you'll see all available namespaces

## Testing Your Setup

### 1. Test the API directly

```bash
# Start the API server
uvicorn context.voice_api:app --reload --port 8000

# Test health endpoint
curl http://localhost:8000/health

# Fetch voice chunks (default: english-sentences namespace)
curl http://localhost:8000/voice-chunks?limit=5

# Fetch from a specific namespace
curl "http://localhost:8000/voice-chunks?limit=5&namespace=english-sentences"

# Fetch from speaker embeddings
curl "http://localhost:8000/voice-chunks?limit=5&namespace=reference&index_name=speaker-embeddings"
```

### 2. Test from Flutter

The Flutter app will automatically use the default semantic search index and namespace. To override:

```dart
// In your Flutter code
VoiceApi.fetchVoiceChunks(
  limit: 10,
  namespace: 'english-sentences',  // Optional: override default
  indexName: 'your-index-name',    // Optional: use different index
);
```

## Common Issues

### Error: "No index configured"
- **Solution**: Make sure `SEMANTIC_INDEX` or `SEMANTIC_INDEX_HOST` is set in `.env`

### Error: "Failed to list vectors"
- **Solution**: Check that:
  1. Your API key is correct
  2. The namespace exists in your index
  3. You have access to the index (check Pinecone dashboard)

### Error: "Namespace not found"
- **Solution**: Verify the namespace name in Pinecone dashboard. Namespaces are case-sensitive!

### Empty results
- **Solution**: The namespace might be empty. Check vector count in Pinecone dashboard.

## Data Structure

### Semantic Search Data (english-sentences namespace)
```json
{
  "id": "182",
  "chunk_text": "May I park here for a while?",
  "lang": "en"
}
```

### Conversation Chunks (conversations namespace)
```json
{
  "id": "chunk-1",
  "text": "User: Did you bring the medicine?",
  "timestamp": "2025-07-05T19:00:00",
  "location": "Home",
  "participants": ["User", "Alex"],
  "tags": ["medicine", "pickup"]
}
```

The API automatically detects which structure you're using and returns the appropriate fields.

