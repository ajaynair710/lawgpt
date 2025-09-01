import json
import re
from tqdm import tqdm
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from rag.rag_pipeline import RAGPipeline


def clean_text(text: str) -> str:
    return text.replace("\n", " ").strip()

def contains_section(text: str) -> bool:
    return bool(re.search(r"\bSection\s+\d+", text, re.IGNORECASE))

def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    return sum(scores) / len(scores)

def compute_f1(preds, refs):
    # Multi-label F1: refs and preds are lists of section numbers
    labels = list(set().union(*refs, *preds))
    y_true = [[1 if l in r else 0 for l in labels] for r in refs]
    y_pred = [[1 if l in p else 0 for l in labels] for p in preds]
    return f1_score(y_true, y_pred, average="micro")



MODEL_NAME = "distilgpt2"   
FINETUNED_DIR = "lawgpt-ipc-lora" 

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")

try:
    ft_model = AutoModelForCausalLM.from_pretrained(FINETUNED_DIR).to("cuda" if torch.cuda.is_available() else "cpu")
except:
    ft_model = None

rag_pipeline = RAGPipeline()


def generate_answer(model, prompt, max_new_tokens=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



def evaluate(eval_set_file="data/eval_set.json", sample_size=50):
    eval_data = json.load(open(eval_set_file, encoding="utf-8"))[:sample_size]

    results = {"base": [], "rag": [], "ft": []}

    rouge_refs, rouge_preds_base, rouge_preds_rag, rouge_preds_ft = [], [], [], []
    f1_refs, f1_preds_base, f1_preds_rag, f1_preds_ft = [], [], [], []
    grounding = {"base": 0, "rag": 0, "ft": 0}

    for ex in tqdm(eval_data, desc="Evaluating"):
        task = ex["task"]  
        prompt = ex["instruction"]

        base_out = generate_answer(base_model, prompt)
        if task == "summarize":
            rouge_refs.append(ex["output"])
            rouge_preds_base.append(base_out)
        elif task == "suggest":
            f1_refs.append(ex["sections"])
            found_sections = re.findall(r"\d+", base_out)
            f1_preds_base.append(found_sections)
        grounding["base"] += int(contains_section(base_out))
        results["base"].append(base_out)

        rag_out = rag_pipeline.query(prompt)
        if task == "summarize":
            rouge_preds_rag.append(rag_out)
        elif task == "suggest":
            found_sections = re.findall(r"\d+", rag_out)
            f1_preds_rag.append(found_sections)
        grounding["rag"] += int(contains_section(rag_out))
        results["rag"].append(rag_out)

        if ft_model:
            ft_out = generate_answer(ft_model, prompt)
            if task == "summarize":
                rouge_preds_ft.append(ft_out)
            elif task == "suggest":
                found_sections = re.findall(r"\d+", ft_out)
                f1_preds_ft.append(found_sections)
            grounding["ft"] += int(contains_section(ft_out))
            results["ft"].append(ft_out)

    rouge_base = compute_rouge(rouge_preds_base, rouge_refs) if rouge_preds_base else 0
    rouge_rag = compute_rouge(rouge_preds_rag, rouge_refs) if rouge_preds_rag else 0
    rouge_ft = compute_rouge(rouge_preds_ft, rouge_refs) if rouge_preds_ft else 0

    f1_base = compute_f1(f1_preds_base, f1_refs) if f1_refs else 0
    f1_rag = compute_f1(f1_preds_rag, f1_refs) if f1_refs else 0
    f1_ft = compute_f1(f1_preds_ft, f1_refs) if f1_refs else 0

    n = len(eval_data)
    results_table = {
        "Model": ["Base", "RAG", "Fine-tuned"],
        "ROUGE-L": [rouge_base, rouge_rag, rouge_ft],
        "F1 (Sections)": [f1_base, f1_rag, f1_ft],
        "Grounding %": [
            grounding["base"]/n*100,
            grounding["rag"]/n*100,
            grounding["ft"]/n*100 if ft_model else 0
        ]
    }

    print("\n Evaluation Results")
    for i in range(3):
        print(f"{results_table['Model'][i]:<10} | "
              f"ROUGE-L: {results_table['ROUGE-L'][i]:.3f} | "
              f"F1: {results_table['F1 (Sections)'][i]:.3f} | "
              f"Grounding: {results_table['Grounding %'][i]:.1f}%")

    return results_table


if __name__ == "__main__":
    evaluate()
