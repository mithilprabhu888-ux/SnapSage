# 🔮 SnapSage

**Snap it. Ask anything.**

SnapSage is a NotebookLM-style AI assistant. Upload your documents (PDF, TXT, MD), ask questions, and get answers **from your files** — streamed live, with citations to the exact source. You can also chat with images, or just chat normally.

Built with **Flask + Groq + Pinecone**.

## What it does

- 📄 **Ask your documents** — upload files, and SnapSage chunks them, embeds them into a Pinecone vector database, and answers your questions using only your content. Every answer cites the file it came from.
- 🖼️ **Ask about images** — attach images (drag & drop, paste, or picker) and ask about them. SnapSage remembers the conversation, so follow-ups work.
- 💬 **Just chat** — with no sources attached, SnapSage is a simple streaming chat assistant.

The UI switches automatically: **DOCS · RAG** mode when sources exist, **CHAT** mode otherwise.

## Setup

```bash
git clone https://github.com/mithilprabhu888-ux/SnapSage.git
cd SnapSage
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env        # then fill in GROQ_API_KEY and PINECONE_API_KEY
python app.py
```

Open http://127.0.0.1:8080. The Pinecone index is created automatically on first run.

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `PINECONE_API_KEY` | *(required for RAG)* | Pinecone API key |
| `PINECONE_INDEX_NAME` | `snapsage-notes` | Vector index name |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Model for document answers |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Vision chat model |
| `PORT` | `8080` | Server port |

Without a Pinecone key, SnapSage still works as a vision chat assistant.

## How RAG works here

1. **Upload** — documents are split into ~150-word overlapping chunks
2. **Embed** — chunks are embedded with `multilingual-e5-large` via Pinecone Inference
3. **Store** — vectors go into a Pinecone serverless index; each browser session gets its own namespace, so notebooks never mix
4. **Ask** — your question is embedded, top-k similar chunks are retrieved, and the LLM answers using *only* that context, citing the source files
5. **No answer in your docs?** — it says *"I couldn't find that in your sources."*
