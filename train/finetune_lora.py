import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "distilgpt2"
OUTPUT_DIR = "lawgpt-ipc-lora"
DATA_FILE = "data/ipc_instructions.json"
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = load_dataset("json", data_files=DATA_FILE, split="train")
print(f" Dataset size: {len(dataset)} samples")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  

def tokenize_function(example):
    text = (
        f"You are a legal assistant. Use only the given IPC sections to answer.\n"
        f"### Instruction:\n{example['instruction']}\n"
        f"### Response:\n{example['output']}"
    )
    return tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH)

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=False,
    remove_columns=dataset.column_names
)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset.select(range(min(200, len(tokenized_dataset)))), 
    tokenizer=tokenizer,
    data_collator=data_collator
)

print(f"🚀 Training on: {DEVICE}")
print(next(model.parameters()).device)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Fine-tuned LoRA model saved to {OUTPUT_DIR}")
