import os
import json
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, request
from openai import OpenAI
from werkzeug.exceptions import RequestEntityTooLarge

from chat_service import ChatService, DEFAULT_MODEL, new_session_id
from rag import RagService, ALLOWED_EXTENSIONS

DIRECTORY = Path(__file__).resolve().parent
load_dotenv(DIRECTORY / ".env")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGES_PER_MESSAGE = 4
SESSION_COOKIE = "snapsage_sid"
SESSION_MAX_AGE = 7 * 24 * 60 * 60  # one week


def get_or_create_sid():
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid:
        sid = new_session_id()
    return sid


def set_session_cookie(response, sid):
    response.set_cookie(
        SESSION_COOKIE, sid,
        max_age=SESSION_MAX_AGE, httponly=True, samesite="Lax")
    return response


def sse_response(generator, sid):
    response = Response(generator, mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return set_session_cookie(response, sid)


def sse_format(event):
    return f"data: {json.dumps(event)}\n\n"


def collect_images():
    """Return uploaded image files, or None when the request has none."""
    files = request.files.getlist("images") or (
        [request.files["image"]] if "image" in request.files else [])
    if not files:
        return None
    if len(files) > MAX_IMAGES_PER_MESSAGE:
        raise ValueError(f"Maximum {MAX_IMAGES_PER_MESSAGE} images per message.")
    for image in files:
        if not image.mimetype.startswith("image/"):
            raise ValueError("Please upload image files only.")
    return files


def build_rag_service(llm_client):
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        return None
    from pinecone import Pinecone
    pc = Pinecone(api_key=api_key)
    index_name = os.getenv("PINECONE_INDEX_NAME", "snapsage-notes")
    llm_model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    return RagService(llm_client, pc, index_name, llm_model)


def create_app(chat_service=None, rag_service=None):
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["MAX_FORM_MEMORY_SIZE"] = MAX_UPLOAD_BYTES

    if chat_service is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GROQ_API_KEY in .env before starting the app.")
        chat_service = ChatService(
            OpenAI(api_key=api_key, base_url=GROQ_BASE_URL,
                   timeout=60.0, max_retries=3),
            os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        )
    if rag_service is None and os.getenv("PINECONE_API_KEY"):
        rag_service = build_rag_service(
            OpenAI(api_key=os.getenv("GROQ_API_KEY"),
                   base_url=GROQ_BASE_URL, timeout=60.0, max_retries=3))

    def text_history(sid):
        """Prior conversation turns that are plain text (vision turns skipped:
        the RAG model is text-only)."""
        session = chat_service._session(sid)
        with session.lock:
            return [m for m in session.history
                    if isinstance(m.get("content"), str)]

    @app.get("/")
    @app.get("/index.html")
    def index():
        response = app.send_static_file("index.html")
        # Always fetch fresh UI on refresh instead of serving a cached page.
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response

    # ------------------------------------------------------------------ #
    # Vision / general chat (unchanged behaviour)
    # ------------------------------------------------------------------ #

    @app.post("/api/chat")
    def chat():
        sid = get_or_create_sid()
        message = request.form.get("message")
        if message is None:
            return Response("The message field is required.", status=400,
                            mimetype="text/plain")
        try:
            images = collect_images()
        except ValueError as error:
            return Response(str(error), status=400, mimetype="text/plain")
        try:
            reply = chat_service.chat(sid, message, images)
        except Exception:
            app.logger.exception("Groq API call failed")
            return Response(
                "Unable to get an AI response. Check your API key and model "
                "name, then try again.",
                status=502, mimetype="text/plain")
        response = Response(
            json.dumps({"reply": reply}), mimetype="application/json")
        return set_session_cookie(response, sid)

    @app.post("/api/chat/stream")
    def chat_stream():
        sid = get_or_create_sid()
        message = request.form.get("message")
        if message is None:
            return Response("The message field is required.", status=400,
                            mimetype="text/plain")
        try:
            images = collect_images()
        except ValueError as error:
            return Response(str(error), status=400, mimetype="text/plain")

        def generate():
            try:
                for delta in chat_service.chat_stream(sid, message, images):
                    yield sse_format({"delta": delta})
                yield sse_format({"done": True})
            except Exception:
                app.logger.exception("Groq streaming call failed")
                yield sse_format({"error": "Unable to get an AI response. "
                                  "Check your API key and model name, then "
                                  "try again."})

        return sse_response(generate(), sid)

    @app.delete("/api/chat")
    def clear_chat():
        sid = get_or_create_sid()
        chat_service.clear_history(sid)
        if rag_service is not None:
            rag_service.clear(sid)
        response = Response(status=200)
        return set_session_cookie(response, sid)

    # ------------------------------------------------------------------ #
    # Notebook sources (NotebookLM-style RAG over uploaded documents)
    # ------------------------------------------------------------------ #

    @app.get("/api/sources")
    def list_sources():
        if rag_service is None:
            return Response(
                "Pinecone is not configured. Set PINECONE_API_KEY in .env.",
                status=503, mimetype="text/plain")
        sid = get_or_create_sid()
        response = Response(
            json.dumps({"sources": rag_service.list_sources(sid)}),
            mimetype="application/json")
        return set_session_cookie(response, sid)

    @app.post("/api/sources")
    def upload_sources():
        if rag_service is None:
            return Response(
                "Pinecone is not configured. Set PINECONE_API_KEY in .env.",
                status=503, mimetype="text/plain")
        sid = get_or_create_sid()
        files = request.files.getlist("files")
        if not files:
            return Response("Attach at least one PDF, TXT or MD file.",
                            status=400, mimetype="text/plain")
        for storage in files:
            name = storage.filename or ""
            ext = Path(name).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                return Response(
                    f"Unsupported file type: {name}. "
                    "Allowed: PDF, TXT, MD.", status=400,
                    mimetype="text/plain")
        try:
            ingested = rag_service.ingest(sid, files)
        except ValueError as error:
            return Response(str(error), status=400, mimetype="text/plain")
        except Exception:
            app.logger.exception("Document ingestion failed")
            return Response("Failed to process the documents. Check your "
                            "Pinecone configuration and try again.",
                            status=502, mimetype="text/plain")
        response = Response(json.dumps({
            "ingested": ingested,
            "sources": rag_service.list_sources(sid),
        }), mimetype="application/json")
        return set_session_cookie(response, sid)

    @app.delete("/api/sources")
    def delete_source():
        if rag_service is None:
            return Response(
                "Pinecone is not configured. Set PINECONE_API_KEY in .env.",
                status=503, mimetype="text/plain")
        sid = get_or_create_sid()
        name = request.args.get("name")
        try:
            if name:
                rag_service.delete_source(sid, name)
            else:
                rag_service.clear(sid)
        except Exception:
            app.logger.exception("Source deletion failed")
            return Response("Failed to delete the source. Try again.",
                            status=502, mimetype="text/plain")
        response = Response(
            json.dumps({"sources": rag_service.list_sources(sid)}),
            mimetype="application/json")
        return set_session_cookie(response, sid)

    @app.post("/api/ask/stream")
    def ask_stream():
        if rag_service is None:
            return Response(
                "Pinecone is not configured. Set PINECONE_API_KEY in .env.",
                status=503, mimetype="text/plain")
        sid = get_or_create_sid()
        question = request.form.get("message")
        if not question or not question.strip():
            return Response("The message field is required.", status=400,
                            mimetype="text/plain")

        def generate():
            for event in rag_service.ask_stream(sid, question.strip(),
                                                text_history(sid)):
                yield sse_format(event)

        return sse_response(generate(), sid)

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(error):
        return Response("Maximum request size is 8 MB.", status=413,
                        mimetype="text/plain")

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.getenv("PORT", "8080")))
