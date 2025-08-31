import json
import random

def build_instruction_dataset(ipc_sections_file: str, output_json: str, max_examples: int = 2500):
    sections = json.load(open(ipc_sections_file, encoding="utf-8"))
    instructions = []

    for sec in sections:
        sec_no, title, text = sec["section_number"], sec["title"], sec["body_text"]

        # Task 1: Explain Section
        instructions.append({
            "instruction": f"Explain IPC Section {sec_no}",
            "input": "",
            "output": f"{text.strip()}"
        })

        # Task 2: Summarize Section
        instructions.append({
            "instruction": f"Summarize IPC Section {sec_no} in simple words",
            "input": "",
            "output": f"{title}. In essence, {text[:200]}..."
        })

        # Task 3: Suggest Sections (template, will mix with handcrafted later)
        instructions.append({
            "instruction": "Suggest IPC sections for a case involving theft and trespassing",
            "input": "",
            "output": "Applicable IPC Sections: 378 (theft), 441 (trespass)."
        })

    random.shuffle(instructions)
    json.dump(instructions[:max_examples], open(output_json, "w", encoding="utf-8"), indent=2)
    print(f"Saved {len(instructions[:max_examples])} instructions → {output_json}")


if __name__ == "__main__":
    build_instruction_dataset("data/ipc_sections.json", "data/ipc_instructions.json")
