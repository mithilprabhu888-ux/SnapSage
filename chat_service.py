"""SnapSage chat service: per-session history, streaming and multi-image vision."""

import base64
import secrets
from threading import Lock

DEFAULT_MODEL = "qwen/qwen3.8-27b"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 800

# Per-session history limits.
MAX_SESSIONS = 100
MAX_HISTORY_MESSAGES = 16

SYSTEM_PROMPT = (
    "You are SnapSage, a friendly AI vision assistant. You answer questions about "
    "images and text. Keep answers short: 3-5 sentences or a few bullet points at "
    "most. Answer only what was asked. No preamble, no summaries, no repeating the "
    "question. If the user shares a screenshot to debug, state the most likely "
    "problem and the exact fix, nothing more. Use markdown (bold, lists, code "
    "blocks) when it helps readability. Give more detail only if the user asks "
    "for it."
)


def new_session_id():
    return secrets.token_urlsafe(24)


class ChatSession:
    """History for one browser session."""

    def __init__(self):
        self.history = []
        self.lock = Lock()


class ChatService:
    """Keeps conversation history per session and talks to a vision LLM."""

    def __init__(self, client, model=DEFAULT_MODEL,
                 temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.sessions = {}
        self.sessions_lock = Lock()

    # ------------------------------------------------------------------ #
    # Session management
    # ------------------------------------------------------------------ #

    def _session(self, session_id):
        with self.sessions_lock:
            if session_id not in self.sessions and len(self.sessions) >= MAX_SESSIONS:
                # Drop the oldest session to bound memory usage.
                oldest = next(iter(self.sessions))
                self.sessions.pop(oldest)
            session = self.sessions.setdefault(session_id, ChatSession())
        return session

    def clear_history(self, session_id):
        session = self._session(session_id)
        with session.lock:
            session.history.clear()

    def _trim(self, history):
        if len(history) > MAX_HISTORY_MESSAGES:
            del history[:-MAX_HISTORY_MESSAGES]

    # ------------------------------------------------------------------ #
    # Message building
    # ------------------------------------------------------------------ #

    def _user_message(self, message, images):
        content = message
        if images:
            parts = [{"type": "text", "text": message}]
            for image in images:
                encoded = base64.b64encode(image.read()).decode("ascii")
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{image.mimetype};base64,{encoded}"},
                })
            content = parts
        return {"role": "user", "content": content}

    def _api_messages(self, session, user_message):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *session.history,
            user_message,
        ]

    # ------------------------------------------------------------------ #
    # Non-streaming chat
    # ------------------------------------------------------------------ #

    def chat(self, session_id, message, images=None):
        session = self._session(session_id)
        with session.lock:
            user_message = self._user_message(message, images)
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=self._api_messages(session, user_message),
            )
            reply = response.choices[0].message.content
            if not isinstance(reply, str):
                raise ValueError("The model returned no text response.")
            # Only commit the turn after the API call succeeds.
            session.history.extend(
                [user_message, {"role": "assistant", "content": reply}])
            self._trim(session.history)
            return reply

    # ------------------------------------------------------------------ #
    # Streaming chat (generator of text deltas)
    # ------------------------------------------------------------------ #

    def chat_stream(self, session_id, message, images=None):
        session = self._session(session_id)
        with session.lock:
            user_message = self._user_message(message, images)
            stream = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=self._api_messages(session, user_message),
                stream=True,
            )
            chunks = []
            try:
                for event in stream:
                    delta = event.choices[0].delta
                    text = getattr(delta, "content", None)
                    if text:
                        chunks.append(text)
                        yield text
            except GeneratorExit:
                raise
            if not chunks:
                raise ValueError("The model returned no text response.")
            # Commit the turn only after the stream completes successfully.
            session.history.extend(
                [user_message, {"role": "assistant", "content": "".join(chunks)}])
            self._trim(session.history)
