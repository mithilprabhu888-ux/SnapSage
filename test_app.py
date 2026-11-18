import copy
import io
import unittest
from types import SimpleNamespace

from app import create_app, MAX_UPLOAD_BYTES
from chat_service import ChatService, DEFAULT_MODEL


def make_client(complete):
    return SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=complete)))


class SnapSageTest(unittest.TestCase):
    def test_chat_image_history_clear_validation_and_failure(self):
        requests = []
        fail = False

        def complete(**request):
            requests.append(copy.deepcopy(request))
            if fail:
                raise RuntimeError("Simulated provider failure")
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="Test reply"))])

        service = ChatService(make_client(complete))
        browser = create_app(service).test_client()

        response = browser.get("/")
        self.assertIn(b"SnapSage", response.data)
        response.close()

        response = browser.post("/api/chat", data={"message": "Hello"})
        self.assertEqual(response.status_code, 200)
        sid = response.headers.get("Set-Cookie", "").split(";")[0].split("=", 1)[1]
        self.assertEqual(requests[0]["model"], DEFAULT_MODEL)
        self.assertEqual(requests[0]["temperature"], 0.4)
        self.assertEqual(requests[0]["messages"][0]["role"], "system")

        response = browser.post("/api/chat", data={
            "message": "Describe it",
            "image": (io.BytesIO(b"image bytes"), "image.png", "image/png"),
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(requests[1]["messages"]), 4)
        content = requests[1]["messages"][3]["content"]
        self.assertEqual(content[0]["text"], "Describe it")
        self.assertEqual(content[1]["image_url"]["url"],
                         "data:image/png;base64,aW1hZ2UgYnl0ZXM=")

        browser.delete("/api/chat")
        browser.post("/api/chat", data={"message": "New conversation"})
        self.assertEqual(len(requests[2]["messages"]), 2)

        self.assertEqual(browser.post("/api/chat", data={}).status_code, 400)
        self.assertEqual(browser.post("/api/chat", data={
            "message": "Bad file",
            "image": (io.BytesIO(b"text"), "file.txt", "text/plain"),
        }).status_code, 400)
        self.assertEqual(browser.post("/api/chat", data={
            "message": "Too large",
            "image": (io.BytesIO(b"x" * (MAX_UPLOAD_BYTES + 1)), "large.png"),
        }).status_code, 413)
        fail = True
        self.assertEqual(browser.post(
            "/api/chat", data={"message": "Failure"}).status_code, 502)
        fail = False

    def test_sessions_are_isolated(self):
        requests = []

        def complete(**request):
            requests.append(copy.deepcopy(request))
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="Reply"))])

        service = ChatService(make_client(complete))
        app = create_app(service)

        # Alice chats.
        alice = app.test_client()
        alice.post("/api/chat", data={"message": "From Alice"})
        self.assertEqual(len(requests[0]["messages"]), 2)

        # A different browser session (fresh client, no cookie) must not
        # see Alice's history.
        bob = app.test_client()
        bob.post("/api/chat", data={"message": "From Bob"})
        self.assertEqual(len(requests[1]["messages"]), 2)
        self.assertEqual(requests[1]["messages"][1]["content"], "From Bob")

        # The same session (cookie kept) continues the conversation.
        response = alice.post("/api/chat", data={"message": "Follow up"})
        self.assertEqual(len(requests[2]["messages"]), 4)

    def test_multi_image_message(self):
        requests = []

        def complete(**request):
            requests.append(copy.deepcopy(request))
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="Reply"))])

        service = ChatService(make_client(complete))
        browser = create_app(service).test_client()
        response = browser.post("/api/chat", data={
            "message": "Compare these",
            "images": [
                (io.BytesIO(b"one"), "a.png", "image/png"),
                (io.BytesIO(b"two"), "b.png", "image/png"),
            ],
        })
        self.assertEqual(response.status_code, 200)
        content = requests[0]["messages"][1]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(len(content), 3)
        self.assertEqual(content[2]["image_url"]["url"].endswith(
            base64_two()), True)

    def test_streaming_endpoint(self):
        def complete(**request):
            def stream():
                for delta in ["Hel", "lo", " world"]:
                    yield SimpleNamespace(choices=[SimpleNamespace(
                        delta=SimpleNamespace(content=delta))])
                yield SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=None))])

            return stream()

        service = ChatService(make_client(complete))
        browser = create_app(service).test_client()
        response = browser.post("/api/chat/stream",
                                data={"message": "Hi"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/event-stream")
        body = response.get_data(as_text=True)
        self.assertIn('"delta": "Hel"', body)
        self.assertIn('"done": true', body)

        # The streamed turn is committed to the session history after
        # completion (sessions hold history now, not the service itself).
        self.assertEqual(len(service.sessions), 1)
        session = next(iter(service.sessions.values()))
        self.assertEqual(len(session.history), 2)
        self.assertEqual(session.history[1]["content"], "Hello world")


def base64_two():
    import base64
    return base64.b64encode(b"two").decode("ascii")


if __name__ == "__main__":
    unittest.main()
