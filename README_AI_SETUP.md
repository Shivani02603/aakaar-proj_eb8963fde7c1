# AI RAG Project Setup Guide

This is an AI-powered Retrieval-Augmented Generation (RAG) application built by Aakaar. It allows users to upload documents and ask questions about them.

## Quick Start (Recommended: Free Tier)

### Step 1: Get a Free Gemini API Key
1. Go to https://aistudio.google.com/apikey
2. Click "Create API Key"
3. Copy the key to your `.env` file (see below)

### Step 2: Setup Environment
```bash
cp .env.example .env
```

Then edit `.env` and set:
```
LLM_API_KEY=AIza...  # Your Gemini API key from step 1
```

For embeddings, the default uses HuggingFace local embeddings (no API key needed — model downloads ~500MB on first run).

### Step 3: Install and Run
```bash
pip install -r requirements.txt
# First time only: downloads the embedding model (~500MB)

# For testing without real API keys:
TESTING=1 pytest tests/

# To run the full app:
uvicorn backend.main:app --reload --port 8000
```

Then open http://localhost:3000 (frontend) and log in.

---

## AI Provider Options

### ✅ Recommended (Free): Gemini LLM + Local Embeddings

**Cost:** $0 (free tier)

**Setup:**
```env
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
RUNTIME_LLM=gemini-2.0-flash
LLM_API_KEY=AIza...   # Free key from https://aistudio.google.com/apikey

# For embeddings, keep defaults (local HuggingFace):
EMBEDDING_MODEL=local-huggingface
```

**Limits:** 60 requests/minute (free tier)  
**Why:** No cost, all features work offline after first model download

---

### Option 2: OpenAI (Paid)

**Cost:** ~$0.01-0.10 per API call

**Setup:**
```env
OPENAI_API_KEY=sk-...  # From https://platform.openai.com/api-keys

RUNTIME_LLM=gpt-4o
LLM_BASE_URL=
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
```

**Limits:** None (pay-as-you-go)  
**Why:** Most reliable, works best for complex queries

---

### Option 3: Groq (Free Alternative)

**Cost:** Free tier available

**Setup:**
```env
LLM_BASE_URL=https://api.groq.com/openai/v1
RUNTIME_LLM=llama-3.3-70b-versatile
LLM_API_KEY=gsk_...  # Get free key at https://console.groq.com

# Groq has NO embeddings API — use Gemini for embeddings:
EMBEDDING_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
EMBEDDING_MODEL=text-embedding-004
EMBEDDING_API_KEY=AIza...  # Gemini API key
```

**Limits:** 30 requests/minute (free tier)  
**Why:** Fast inference, completely free tier available

---

### Option 4: All Gemini (Both LLM and Embeddings)

**Cost:** Free tier

**Setup:**
```env
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
RUNTIME_LLM=gemini-2.0-flash
LLM_API_KEY=AIza...

EMBEDDING_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
EMBEDDING_MODEL=text-embedding-004
EMBEDDING_API_KEY=AIza...  # Same key works for both
```

**Limits:** 60 requests/minute (free tier)  
**Why:** Single API key, everything cloud-based, no local downloads

---

## Database Setup

By default, PostgreSQL is required. For development, use SQLite (automatic with `TESTING=1`):

```bash
# Development with SQLite:
TESTING=1 pytest tests/
TESTING=1 uvicorn backend.main:app --reload

# Production with PostgreSQL:
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
uvicorn backend.main:app --reload
```

---

## Running Tests

Tests run hermetically with TESTING=1 (no real API keys or DB server needed):

```bash
TESTING=1 pytest tests/ -v
```

What's tested:
- Document upload and ingestion
- Semantic search over embedded chunks
- Chat history and streaming responses
- User isolation (can't see other users' documents)

---

## Troubleshooting

### "EMBEDDING_API_KEY not found"
You're using a cloud embedding provider but forgot to set the API key.  
**Fix:** Either set `EMBEDDING_API_KEY` in `.env`, or switch to `EMBEDDING_MODEL=local-huggingface`

### "LLM_API_KEY and OPENAI_API_KEY are both missing"
The LLM provider can't find credentials.  
**Fix:** Set either `LLM_API_KEY` or `OPENAI_API_KEY` in `.env`, or specify which provider in `LLM_BASE_URL` and `RUNTIME_LLM`

### Slow embeddings on first run
The local HuggingFace model is downloading (~500MB).  
**Fix:** Wait, it's one-time only. Or switch to a cloud provider.

### Chat responses are gibberish
The LLM provider is misconfigured or the model name is wrong.  
**Fix:** 
- Verify `LLM_BASE_URL` and `RUNTIME_LLM` match your provider
- Check `LLM_API_KEY` is valid
- Check the provider accepts that model name

### Tests fail with "sqlite3 database is locked"
Multiple test workers are hitting the same SQLite DB.  
**Fix:** Run tests serially: `TESTING=1 pytest tests/ -n0`

---

## Deployment

### Docker
```bash
docker build -t my-rag-app .
docker run -e GEMINI_API_KEY=AIza... -p 8000:8000 my-rag-app
```

### Environment Variables Required
- `DATABASE_URL` — production database
- `JWT_SECRET` — random string for auth tokens
- `LLM_API_KEY` or `OPENAI_API_KEY` — AI provider credentials
- `EMBEDDING_API_KEY` (if not using local embeddings)
- `FRONTEND_URL` / `VITE_API_URL` — for CORS

See `.env.example` for all options.

---

## Architecture

- **Backend:** FastAPI with SQLAlchemy ORM
- **Frontend:** Next.js + React
- **Embeddings:** Local HuggingFace (`all-mpnet-base-v2`) or cloud (OpenAI/Gemini)
- **LLM:** Configurable provider (Gemini/OpenAI/Groq)
- **Vector Store:** PostgreSQL with pgvector (or SQLite in dev)

---

## Free Tier Recommendations

| Component | Provider | Cost | Setup Time |
|-----------|----------|------|-----------|
| LLM Chat | **Gemini** | Free | 2 min |
| Embeddings | **Local HuggingFace** | Free | 5 min (first download) |
| Database | PostgreSQL Cloud (free tier) | Free | 5 min |
| **Total** | | **$0/month** | **12 min** |

No credit card required. Start building!
