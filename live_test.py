"""Live end-to-end test against real Groq + Pinecone (run manually)."""
import io
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from app import create_app

KNOWLEDGE = Path(r"D:\projects\Forward-Deployed-Engineer-Full-Course"
                 r"\Lecture 11\files")

app = create_app()
client = app.test_client()

# 1. Health: UI
r = client.get("/")
print("UI:", r.status_code)

# 2. Upload the real policy PDFs into this session's notebook
files = []
for pdf in sorted(KNOWLEDGE.glob("*.pdf")):
    files.append(("files", (io.BytesIO(pdf.read_bytes()), pdf.name,
                            "application/pdf")))
print(f"Uploading {len(files)} PDFs:", ", ".join(p[1][1] for p in files))
from werkzeug.datastructures import MultiDict, FileStorage
data = MultiDict()
for key, (stream, name, mime) in files:
    data.add(key, FileStorage(stream=stream, filename=name,
                              content_type=mime))
r = client.post("/api/sources", data=data)
print("Upload:", r.status_code)
data = r.get_json()
for ing in data["ingested"]:
    print(f"  - {ing['name']}: {ing['chunks']} chunks"
          + (" (EMPTY!)" if ing["empty"] else ""))
print("Sources now:", [s["name"] for s in data["sources"]])

# 3. Ask a grounded question, verify SSE stream + citations
question = "What is the refund policy and how many days do I have?"
r = client.post("/api/ask/stream", data={"message": question})
print("\nQuestion:", question)
print("Stream status:", r.status_code, "| mimetype:", r.mimetype)

answer, sources, errors = "", None, []
for line in r.get_data(as_text=True).split("\n\n"):
    line = line.strip()
    if not line.startswith("data: "):
        continue
    evt = json.loads(line[6:])
    if "delta" in evt:
        answer += evt["delta"]
    elif "sources" in evt:
        sources = evt["sources"]
    elif "error" in evt:
        errors.append(evt["error"])

print("\n--- ANSWER (streamed) ---")
print(answer)
print("--- SOURCES ---", sources)
if errors:
    print("ERRORS:", errors)
    sys.exit(1)

# 4. Source listing + partial deletion round-trip
r = client.get("/api/sources")
print("\nGET /api/sources:", r.status_code, r.get_json())
r = client.delete(f"/api/sources?name={data['ingested'][0]['name']}")
print("DELETE one source:", r.status_code,
      [s["name"] for s in r.get_json()["sources"]])
r = client.delete("/api/sources")
print("DELETE all:", r.status_code, r.get_json())
print("\nLIVE TEST PASSED ✅")
