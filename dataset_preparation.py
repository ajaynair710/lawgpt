import pdfplumber
import re
import json
import pandas as pd
import random


# Step 1: Extract raw text from PDF

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


# Step 2: Parse IPC sections

def parse_ipc_sections(raw_text):
    pattern = re.compile(r"^(\d+[A-Z]?)\.\s*(.*)")

    sections = []
    current_section = None

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        match = pattern.match(line)
        if match:
            if current_section:
                sections.append(current_section)

            section_number = match.group(1)
            title = match.group(2).strip()
            current_section = {
                "section_number": section_number,
                "title": title,
                "body_text": ""
            }
        else:
            if current_section:
                current_section["body_text"] += " " + line

    if current_section:
        sections.append(current_section)

    return sections


# Step 3: Save JSON & CSV

def save_data(sections, json_file, csv_file):
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(sections)
    df.to_csv(csv_file, index=False, encoding="utf-8")


# Step 4: Generate Instruction Dataset

def generate_instruction_dataset(sections, out_json="ipc_instructions.json"):
    dataset = []

    for sec in sections:
        number = sec["section_number"]
        title = sec["title"]
        body = sec["body_text"].strip()

        if not body:
            continue

        # 1. Explain Section
        dataset.append({
            "input": f"Explain IPC Section {number}",
            "output": f"Section {number} {title}. {body}"
        })

        # 2. Summarize Section
        dataset.append({
            "input": f"Summarize IPC Section {number} in simple words",
            "output": f"Section {number} is about {title.lower()}."
        })

        # 3. Suggest Section (create synthetic case examples randomly)
        if random.random() < 0.3: 
            case_text = f"A case related to {title.lower()}."
            dataset.append({
                "input": case_text,
                "output": f"Applicable IPC Section: {number} ({title})"
            })

    # Save dataset
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    return dataset


# MAIN PIPELINE

if __name__ == "__main__":
    pdf_path = "data/indian_ipc.pdf"
    json_file = "data/res/ipc_sections.json"
    csv_file = "data/res/ipc_sections.csv"
    instr_file = "data/res/ipc_instructions.json"

    print("Extracting text...")
    raw_text = extract_text_from_pdf(pdf_path)

    print("Parsing sections...")
    sections = parse_ipc_sections(raw_text)

    print(f"Parsed {len(sections)} sections")

    print("Saving structured data...")
    save_data(sections, json_file, csv_file)

    print("Generating instruction dataset...")
    dataset = generate_instruction_dataset(sections, instr_file)

    print(f"Generated {len(dataset)} instruction samples")
