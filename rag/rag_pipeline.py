import faiss
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

class RAGPipeline:
    def __init__(self, index_file="rag/ipc.index", sections_file="rag/ipc_sections.json"):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = faiss.read_index(index_file)
        self.sections = json.load(open(sections_file, encoding="utf-8"))

        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm = AutoModelForCausalLM.from_pretrained("distilgpt2").to(self.device)

    def _clean_and_format_sections(self, retrieved, max_sections=5):
        seen = set()
        clean_sections = []
        for s in retrieved:
            sec_num = s["section_number"]
            if sec_num in seen:
                continue
            seen.add(sec_num)
            title = s["title"].strip()
            body = s["body_text"].strip().replace("\n", " ")
            clean_sections.append(f"Section {sec_num}: {title}. {body}")
            if len(clean_sections) >= max_sections:
                break
        return "\n".join(clean_sections)

    def query(self, user_query, top_k=5, max_context_tokens=512, max_new_tokens=300):
        embedding = self.model.encode([user_query])
        D, I = self.index.search(embedding, top_k)
        retrieved = [self.sections[i] for i in I[0]]

        context = self._clean_and_format_sections(retrieved)

        prompt = (
            f"You are a legal assistant. Answer the query using ONLY the IPC sections below.\n"
            f"- Explain briefly.\n"
            f"- Mention relevant section numbers like Section 420, Section 441, etc.\n\n"
            f"Query: {user_query}\n\n"
            f"IPC Sections:\n{context}\n\nAnswer:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_context_tokens
        ).to(self.device)

        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=3,
            do_sample=False
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = re.sub(r'(\bSection \d+:)\s*\1+', r'\1', answer)
        return answer

if __name__ == "__main__":
    rag = RAGPipeline()
    query = "A person entered a house at night with intent to commit theft."
    print(rag.query(query))
    query2 = "What sections apply to a case of cheating and forgery?"
    print(rag.query(query2))