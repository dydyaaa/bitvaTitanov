import json
import re

from docx import Document


def parse_glossary(docx_file, json_file):
    doc = Document(docx_file)
    glossary = {}

    term = None
    definition_parts = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Если в абзаце есть жирный текст → это термин
        if any(run.bold for run in para.runs):
            # Сохраняем предыдущий термин
            if term and definition_parts:
                definition = "\n".join(definition_parts).strip()
                definition = re.sub(r'"(.*?)"', r'«\1»', definition)
                glossary[term] = definition
                definition_parts = []

            # Берём сам термин (убираем номера "26.3.9.")
            term_text = re.sub(r"^\d+(\.\d+)*\.?\s*", "", text)
            term = term_text
        else:
            if term:
                definition_parts.append(text)

    # Сохраняем последний термин
    if term and definition_parts:
        definition = "\n".join(definition_parts).strip()
        definition = re.sub(r'"(.*?)"', r'«\1»', definition)
        glossary[term] = definition

    # Запись в JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)


# Пример использования
parse_glossary("glossariy_text.docx", "glossary.json")
