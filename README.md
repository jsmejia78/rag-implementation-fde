# YouTube Transcript RAG

**`001. YouTube RAG.ipynb`**

<a href="https://colab.research.google.com/github/jsmejia78/rag-implementation-fde/blob/main/001.%20YouTube%20RAG.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

A simple Retrieval-Augmented Generation (RAG) pipeline using semantic search. It takes a YouTube video, uses its transcript as the knowledge source, and answers questions about the video with citations back to the transcript.

The task this notebook solves is described in [TASK.md](TASK.md).

## How it works

```
        YouTube video
              │
              ▼
   1. Extract transcript ──────► TRANSCRIPT.md (cached)
              │
              ▼
   2. Split into chunks          naive fixed-size, 25% overlap
              │
              ▼
   3. Generate embeddings        OpenAI text-embedding-3-small
              │
              ▼
   4. Store in Qdrant            local vector store (qdrant_data/)


        User query
              │
              ▼
   5. Retrieve top-3 chunks      embed query → Qdrant similarity search
              │
              ▼
   6. Generate answer            OpenAI chat model, grounded in the chunks
              │
              ▼
     Answer with citations [1][2]
```

## Key components

| Function | Role |
|---|---|
| `get_video_title()` / `fetch_transcript()` | Look up the video title and download its transcript |
| `load_cached_transcript()` | Reuses `TRANSCRIPT.md` when it exists, has content, and its title matches the video |
| `chunk_text()` | Naive fixed-size chunking with a configurable overlap |
| `get_text_embeddings()` | Converts chunks (or a query) into 1536-dim vectors with `text-embedding-3-small` |
| `retrieve_chunks()` | Embeds the query and returns the top-k closest chunks from Qdrant |
| `rag_formatted_response()` | Asks the chat model to answer using only the retrieved chunks, with `[1][2]` citations |
| `youtube_rag()` | Runs retrieval and generation in one call |

## Settings you can change

| Variable | Default | Meaning |
|---|---|---|
| `VIDEO_URL` | an MLOps vs LLMOps video | The YouTube video used as the knowledge source |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP_PCT` | `25` | Overlap between consecutive chunks, as a % of the chunk size |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `TOP_K` | `3` | Chunks retrieved per query |
| `GENERATION_MODEL` | `gpt-5.6-luna` | OpenAI chat model that writes the answer |

## Running it

You need an OpenAI API key.

### Locally (VS Code / Jupyter)

```bash
uv sync                      # creates .venv from pyproject.toml
cp .env.example .env         # then put your key in .env
```

Open the notebook, select the `.venv` interpreter as the kernel, and run all cells. Without uv, `pip install -r requirements.txt` works too.

### On Google Colab

Open the notebook with the badge above and add `OPENAI_API_KEY` in the 🔑 Secrets panel. The first cells install the libraries and read the key from there.

YouTube often blocks transcript requests coming from cloud IPs. If the extraction step fails on Colab, upload `TRANSCRIPT.md` from this repo next to the notebook and the saved transcript will be used instead.

## Files

| File | What it is |
|---|---|
| `001. YouTube RAG.ipynb` | The notebook with the full pipeline |
| `TRANSCRIPT.md` | Cached transcript of the current video (title on the first line) |
| `TASK.md` | The task description |
| `pyproject.toml` / `uv.lock` / `requirements.txt` | Python dependencies |
| `.env.example` | Template for the local `.env` holding the API key |

The local Qdrant store (`qdrant_data/`) is rebuilt on every run, so it is not committed.
