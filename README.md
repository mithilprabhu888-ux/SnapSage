# 🔮 SnapSage

**Snap it. Ask anything.** A NotebookLM-style AI notebook + vision chat assistant:

- **📄 Document RAG** — upload PDFs / TXT / MD files, and ask questions about them. Chunks are embedded with Pinecone's `multilingual-e5-large` model and stored in a **Pinecone vector database**; answers are retrieved (top-k vector search), grounded in the matching chunks, streamed live, and **cited with the source filename**.
- **🖼️ Vision chat** — attach images (drag & drop, paste, picker) and ask about them, multi-turn, via Groq's vision models.

Built with **Flask + Groq + Pinecone**, evolved from the Lecture 16 (vision chatbot) and Lecture 11 (RAG shop-support) projects.

## Features

### Notebook (RAG over your documents)
- Upload up to 25 sources per notebook (PDF, TXT, MD)
- Word-based chunking (150 words / 30 overlap) → Pinecone Inference embeddings → vector upsert
- Each browser session gets its **own notebook namespace** — documents never mix between users
- Streaming answers (SSE) with `[filename.pdf]` citations and source chips under each answer
- Grounded prompts: the model may only use your documents, and says *"I couldn't find that in your sources."* otherwise
- Per-source removal and clear-all (vectors deleted from Pinecone by metadata filter)
- Conversation-aware: recent text turns are included so follow-up questions work

### Vision chat (unchanged SnapSage core)
- Multi-turn vision with up to 4 images per message
- Client-side image resizing before upload, session-isolated history
- The UI switches mode automatically: **DOCS · RAG** when sources exist, **CHAT** otherwise (images always force vision chat)

## Setup

```bash
cd SnapSage
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env        # then fill in GROQ_API_KEY and PINECONE_API_KEY
python app.py
```

Open http://127.0.0.1:8080. The Pinecone index (`snapsage-notes` by default) is
created automatically on first run (AWS us-east-1 serverless).

## Configuration (all via `.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key (chat + answers) |
| `PINECONE_API_KEY` | *(required for RAG)* | Pinecone API key |
| `PINECONE_INDEX_NAME` | `snapsage-notes` | Vector index name |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Model used for grounded answers |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Vision chat model |
| `PORT` | `8080` | Server port |

Without `PINECONE_API_KEY` the app still works as a vision chat assistant
(document endpoints return `503`).

## API

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/chat` | Vision/general chat (+ optional `images`), returns JSON `{reply}` |
| `POST` | `/api/chat/stream` | Same, streamed as SSE |
| `POST` | `/api/sources` | Upload `files` (PDF/TXT/MD) → chunk, embed, store; returns `{ingested, sources}` |
| `GET` | `/api/sources` | List this notebook's sources `{sources: [{name, chunks}]}` |
| `DELETE` | `/api/sources?name=` | Remove one source (or all without `name`) |
| `POST` | `/api/ask/stream` | RAG question over your sources, streamed SSE with a final `sources` event |
| `DELETE` | `/api/chat` | Clear this session's chat history **and** notebook |

SSE format: `data: {"delta": "..."}` chunks; vision chat ends with `{"done": true}`,
RAG ends with `{"sources": ["file.pdf", ...]}`; errors arrive as `{"error": "..."}`.

## Tests

```bash
python -m unittest discover -p "test_*.py" -v
```

11 tests cover the vision chat, session isolation, streaming, PDF/TXT ingestion,
grounded answers, citations, source deletion and the no-Pinecone fallback.

## Notes & limits

- Uploads capped at **8 MB per request**; images are pre-resized client-side
- History/chunks are in-memory manifests per process; vector data lives in Pinecone
  (restarting the server clears chat + the source list, vectors remain until deleted)
- Embeddings use Pinecone Inference (`multilingual-e5-large`, 1024 dims, cosine)
