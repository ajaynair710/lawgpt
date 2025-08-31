# 📜 Law-GPT: Legal Language Model for Indian Penal Code (IPC)

**Law-GPT** is a lightweight, transformer-based language model fine-tuned to assist with queries about the Indian Penal Code (IPC). It can:

* Explain IPC sections in simple language.
* Summarize sections.
* Suggest relevant IPC sections for legal cases.

This project demonstrates **LLM fine-tuning (LoRA)**, **RAG pipelines**, and **end-to-end evaluation** — optimized to run on local hardware.

---

## 🔹 Project Structure

```
law-gpt-ipc/
│── data/
│   ├── ipc.pdf                 # Official IPC PDF
│   ├── ipc_parser.py           # Extract sections from PDF
│   ├── ipc_sections.json       # Structured IPC sections
│   ├── ipc_instructions.py     # Instruction dataset generator
│   ├── ipc_instructions.json   # Instruction dataset
│   └── eval_set.json           # Small evaluation set
│
│── rag/
│   ├── rag_pipeline.py         # RAG inference pipeline
│
│── train/
│   ├── finetune_lora.py # Fine-tune LoRA on small model
│
│── eval/
│   ├── evaluate_models.py       # Evaluate Base, RAG, Fine-tuned
│
│── app/
│   ├── app.py                   # Gradio demo app
│
│── requirements.txt
│── README.md
```

---

## 🔹 Installation

1. Clone the repo:

```bash
git clone https://github.com/ajaynair710/lawgpt.git
cd lawgpt
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔹 Dataset Preparation

1. Place official IPC PDF in `data/ipc.pdf`.
2. Extract structured sections:

```bash
python data/ipc_parser.py
```

3. Generate instruction dataset (\~2k examples):

```bash
python data/ipc_instructions.py
```

---

## 🔹 RAG Pipeline

Build embeddings and perform retrieval + generation:

```bash
python rag/rag_pipeline.py
```

Example query:

```text
A person forcibly entered someone's house at night with intent to harm. Which IPC sections may apply?
```

---

## 🔹 Fine-Tuning

Fine-tune a **small GPT model (distilgpt2)** using LoRA:

```bash
python train/train_finetune_small.py
```

* Adapter weights saved in `lawgpt-ipc-lora/`
* Small model = fast local training (<30 min CPU, <10 min GPU). 
* I have done CPU training on my local.

---

## 🔹 Evaluation

Evaluate **Base**, **RAG**, and **Fine-Tuned** models:

```bash
python eval/evaluate_models.py
```

Metrics computed:

* **ROUGE-L** → for summarization/explanation
* **F1** → for suggested sections
* **Grounding %** → proportion of outputs mentioning valid section numbers

Example results (small model):

```
Base       | ROUGE-L: 0.168 | F1: 0.000 | Grounding: 80.0%
RAG        | ROUGE-L: 0.030 | F1: 0.165 | Grounding: 100.0%
Fine-tuned | ROUGE-L: 0.059 | F1: 0.000 | Grounding: 80.0%
```

> Note: Small model used for local demo. Larger models (Mistral-7B) will significantly improve performance.

---

## 🔹 Demo App

Launch a Gradio demo:

```bash
python app/app.py
```

* Input your query and choose:

  * Base
  * RAG
  * Fine-tuned
* Outputs include **cited IPC sections**.
* Can be deployed to **Hugging Face Spaces**.

---

## 🔹 Usage Examples

```python
from rag.rag_pipeline import rag_query

query = "A person committed theft and trespassed a house."
answer = rag_query(query)
print(answer)
```

```python
# Fine-tuned inference
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_dir = "lawgpt-ipc-lora"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir).to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "Explain IPC Section 420"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🔹 Key Features

* **Instruction Fine-Tuning (LoRA)** → small footprint, fast training.
* **RAG Pipeline** → retrieval + generation improves grounding.
* **Evaluation Suite** → ROUGE, F1, grounding metrics.
* **Demo Web App** → interactive query interface.

---

## 🔹 Future Improvements

* Use **larger LLMs** (Mistral/Falcon) for better quality.
* Expand instruction dataset to 10k+ examples.
* Add **multi-turn legal QA** support.
* Combine RAG + fine-tuned model for **hybrid approach**.

---

## 🔹 Disclaimer

**Law-GPT is for educational/demo purposes only. It does not provide legal advice. Users should consult qualified legal professionals.**

---

