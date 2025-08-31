import re
import json
from pathlib import Path
import fitz  

def extract_ipc_sections(pdf_path: str, output_json: str):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")

    section_pattern = re.compile(r"^(\d+[A-Z]?)\.\s*(.*)")

    sections = []
    current_section = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        match = section_pattern.match(line)
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

    Path(output_json).write_text(json.dumps(sections, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Extracted {len(sections)} sections → {output_json}")


if __name__ == "__main__":
    extract_ipc_sections("data/ipc.pdf", "data/ipc_sections.json")
