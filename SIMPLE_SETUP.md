# Simple Setup Guide

## Why We Need a Backend (FastAPI)

**You can't put Pinecone API keys directly in Flutter apps** - they would be exposed and anyone could access your database. The FastAPI backend keeps your keys secure on the server.

## Super Simple Setup

### 1. Create `.env` file in project root:

```bash
PINECONE_API_KEY=your-api-key-here
PINECONE_INDEX_HOST=https://speaker-embeddings-huei5j9.svc.aped-4627-b74a.pinecone.io
PINECONE_NAMESPACE=reference
```

That's it! The index host is already set as default in the code, so you only need to set it if it's different.

### 2. Start the API:

```bash
uvicorn context.voice_api:app --reload --port 8000
```

### 3. Test it:

```bash
curl http://localhost:8000/speakers
```

You should see all your speakers (Sarah, Sforachel, UnknownSpeaker20250922_023036, etc.)

### 4. Run Flutter app:

The app will automatically fetch from `http://localhost:8000/speakers` (or `http://10.0.2.2:8000` on Android emulator)

## What Changed

- **Before**: Complex API with multiple indexes, namespaces, query params
- **Now**: One simple endpoint `/speakers` that fetches from your speaker embeddings index

The API is now just 70 lines instead of 146 lines!

## API Endpoints

- `GET /health` - Check if API is running
- `GET /speakers?limit=50` - Get all speakers from Pinecone

That's it! Simple and direct.

