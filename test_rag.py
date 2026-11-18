"""Tests for the NotebookLM-style RAG flow using a fake Pinecone client."""

import io
import unittest
from types import SimpleNamespace

from rag import RagService


class FakeEmbedResult:
    def __init__(self, dim=8):
        self.data = [{"values": [0.1] * dim} for _ in range(1)]


class FakeInference:
    def embed(self, model, inputs, parameters):
        return SimpleNamespace(data=[{"values": [0.1, 0.2]} for _ in inputs])


class FakeIndex:
    def __init__(self):
        self.vectors = {}   # namespace -> {id: (values, metadata)}

    def upsert(self, vectors, namespace):
        store = self.vectors.setdefault(namespace, {})
        for v in vectors:
            store[v["id"]] = (v["values"], v["metadata"])

    def query(self, vector, top_k, include_metadata, namespace):
        store = self.vectors.get(namespace, {})
        matches = [SimpleNamespace(id=i, score=0.9, metadata=meta)
                   for i, (_, meta) in store.items()]
        return SimpleNamespace(matches=matches[:top_k])

    def delete(self, namespace, delete_all=False, filter=None):
        store = self.vectors.get(namespace)
        if store is None:
            return
        if delete_all:
            store.clear()
        elif filter:
            source = filter.get("source", {}).get("$eq")
            for i in [i for i, (_, meta) in store.items()
                      if meta.get("source") == source]:
                del store[i]


class FakePinecone:
    def __init__(self):
        self.indexes = set()
        self.inference = FakeInference()
        self._index = FakeIndex()

    def list_indexes(self):
        return SimpleNamespace(names=lambda: list(self.indexes))

    def create_index(self, name, dimension, metric, spec):
        self.indexes.add(name)

    def Index(self, name):
        return self._index


def make_llm(chunks, fail=False):
    def complete(**request):
        if fail:
            raise RuntimeError("Simulated provider failure")

        def stream():
            for delta in chunks:
                yield SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=delta))])

        return stream()

    return SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=complete)))


class RagServiceTest(unittest.TestCase):
    def make_service(self, llm=None):
        pc = FakePinecone()
        service = RagService(llm or make_llm(["Answer"]), pc,
                             "snapsage-notes", "llama-3.3-70b-versatile")
        return service, pc

    @staticmethod
    def make_pdf(text):
        """Build a small but valid PDF (via pypdf's own writer)."""
        from pypdf import PdfWriter
        from pypdf.generic import (DictionaryObject, NameObject, FloatObject,
                                   StreamObject, ArrayObject)
        writer = PdfWriter()
        page = writer.add_blank_page(612, 792)
        content = StreamObject()
        content.set_data(
            b"BT /F1 12 Tf 72 720 Td (" + text.encode("ascii") + b") Tj ET")
        page[NameObject("/Contents")] = writer._add_object(content)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        font_ref = writer._add_object(font)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({
                NameObject("/F1"): font_ref,
            }),
            NameObject("/MediaBox"): ArrayObject(
                [FloatObject(v) for v in (0, 0, 612, 792)]),
        })
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()

    def upload(self, service, sid, name, content):
        storage = SimpleNamespace(filename=name,
                                  read=lambda content=content: content)
        return service.ingest(sid, [storage])

    def test_ingest_chunks_upsert_and_list(self):
        service, _ = self.make_service()
        text = (" ".join(["word"] * 200)).encode("utf-8")
        result = self.upload(service, "sid1", "notes.txt", text)
        self.assertEqual(result[0]["chunks"], 2)   # 150 words + 50 overlap tail
        sources = service.list_sources("sid1")
        self.assertEqual(sources, [{"name": "notes.txt", "chunks": 2}])

        # Two ingests of the same file accumulate counts.
        self.upload(service, "sid1", "notes.txt", text)
        self.assertEqual(service.list_sources("sid1")[0]["chunks"], 4)

    def test_pdf_extraction(self):
        service, _ = self.make_service()
        pdf = self.make_pdf("Hello SnapSage. " * 30)
        result = self.upload(service, "sid1", "doc.pdf", pdf)
        self.assertFalse(result[0]["empty"])
        self.assertGreaterEqual(result[0]["chunks"], 1)

    def test_empty_file_is_flagged(self):
        service, _ = self.make_service()
        result = self.upload(service, "sid1", "empty.txt", b"   ")
        self.assertTrue(result[0]["empty"])
        self.assertEqual(service.list_sources("sid1"), [])

    def test_ask_stream_grounded_with_sources(self):
        service, _ = self.make_service(make_llm(["The policy ", "is 30 days."]))
        self.upload(service, "sid1", "policy.txt",
                    b"The refund window is 30 days. " * 30)
        events = list(service.ask_stream("sid1", "What is the refund window?"))
        deltas = [e["delta"] for e in events if "delta" in e]
        self.assertEqual("".join(deltas), "The policy is 30 days.")
        sources = [e["sources"] for e in events if "sources" in e]
        self.assertEqual(sources, [["policy.txt"]])

    def test_ask_stream_no_matches(self):
        service, _ = self.make_service()
        events = list(service.ask_stream("sid1", "Anything?"))
        self.assertEqual(events[0]["delta"],
                         "I couldn't find that in your sources.")
        self.assertEqual(events[1]["sources"], [])

    def test_history_is_text_only_and_in_prompt(self):
        service, _ = self.make_service(make_llm(["Reply"]))
        self.upload(service, "sid1", "doc.txt", b"content " * 100)
        history = [
            {"role": "user", "content": [{"type": "text", "text": "with image"}]},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "plain question"},
        ]
        events = list(service.ask_stream("sid1", "next question", history))
        self.assertIn("delta", events[0])
        self.assertEqual(events[-1]["sources"], ["doc.txt"])

    def test_delete_source_and_clear(self):
        service, pc = self.make_service()
        self.upload(service, "sid1", "a.txt", b"alpha " * 100)
        self.upload(service, "sid1", "b.txt", b"beta " * 100)

        service.delete_source("sid1", "a.txt")
        self.assertEqual([s["name"] for s in service.list_sources("sid1")],
                         ["b.txt"])

        service.clear("sid1")
        self.assertEqual(service.list_sources("sid1"), [])
        self.assertEqual(pc._index.vectors.get(service._namespace("sid1")), {})


if __name__ == "__main__":
    unittest.main()
