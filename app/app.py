import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rag.rag_pipeline import RAGPipeline

# ---------- Load RAG ----------
rag = RAGPipeline()

# ---------- Load Base Model ----------
BASE_MODEL_NAME = "distilgpt2"
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model.to(device)

# ---------- Load Fine-tuned Model ----------
FT_MODEL_DIR = "lawgpt-ipc-lora"
try:
    ft_tokenizer = AutoTokenizer.from_pretrained(FT_MODEL_DIR)
    ft_model = AutoModelForCausalLM.from_pretrained(FT_MODEL_DIR)
    ft_model.to(device)
except Exception as e:
    print("Could not load fine-tuned model:", e)
    ft_model = None
    ft_tokenizer = base_tokenizer

# ---------- Generation Function ----------
def generate_answer(model, tokenizer, query, max_new_tokens=200):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------- Gradio Function ----------
def answer(query, mode):
    if mode == "RAG Baseline":
        return rag.query(query)
    elif mode == "Fine-tuned" and ft_model is not None:
        return generate_answer(ft_model, ft_tokenizer, query)
    else:
        return generate_answer(base_model, base_tokenizer, query)

# ---------- Gradio Interface ----------
iface = gr.Interface(
    fn=answer,
    inputs=[
        gr.Textbox(label="Your Query", placeholder="Describe your case or ask about IPC sections..."),
        gr.Radio(["RAG Baseline", "Fine-tuned", "Base"], value="RAG Baseline")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Law-GPT (IPC)",
    description="Experimental IPC legal assistant. Not legal advice."
)

if __name__ == "__main__":
    iface.launch()
