from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

def build_faiss_index(ipc_file="data/ipc_sections.json", index_file="rag/ipc.index"):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sections = json.load(open(ipc_file, encoding="utf-8"))

    texts = [f"Section {s['section_number']}: {s['title']} {s['body_text']}" for s in sections]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_file)
    json.dump(sections, open("rag/ipc_sections.json", "w", encoding="utf-8"), indent=2)
    print(f"Built FAISS index with {len(sections)} sections.")

if __name__ == "__main__":
    build_faiss_index()
