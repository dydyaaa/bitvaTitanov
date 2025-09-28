import os
import re
from uuid import uuid4

import chromadb
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PATH = "src/chroma_db/"
PDF_DIR = "src/docs/"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
COLLECTION = "rzd_docs"

# Минимальная длина содержательного чанка
MIN_CHUNK_CHARS = 80

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""]
)

all_docs: list[Document] = []

# ====== Регэкспы и ключевые слова для метаданных ======
SECTION_RE = re.compile(r'(?:Раздел|Глава)\s+([IVXLC\d]+)[\.\:\)]?\s*(.+)?', re.IGNORECASE)
CLAUSE_RE  = re.compile(r'(?:п\.|пункт)\s*([\d]+(?:\.[\d]+)*)', re.IGNORECASE)

KW_LIST = [
    "зазор",
    "консульт",
    "заменяют",
    "сто ржд",
    "электроу",
    "проверит",
    "более",
    "сохранен",
    "утверждении вводе",
    "ржд 2021",
    "действие сто",
    "вводе действие",
    "консультантплюс дата",
    "дата сохранения",
    "управления документ",
    "правовая поддержка",
    "ремонту железнодорожного",
    "состава механизмов",
    "проверяе",
    "распоряжение оао",
    "дата",
    "страница",
    "система управления",
    "вала",
    "вводе",
    "предоставлен консультантплюс",
    "предоста",
    "документ предоставлен",
    "знаний",
    "электроб",
    "осмотрет",
    "распоряж",
    "втулки",
    "оао ржд",
    "дизеля",
    "электрох",
    "трещины",
    "настоящего руководства",
    "подразде",
    "обслуживанию ремонту",
    "обработк",
    "персонал",
    "трещин",
    "документ",
    "управлен",
    "корпуса",
    "допускае",
    "насоса",
    "птээп",
    "шестерен",
    "автомати",
    "очищаютс",
    "трудовых функций",
    "заменяет",
    "разрешае",
    "поверхно",
    "железнодорожного подвижного",
    "контактной сети",
    "части регламентирующей",
    "регламентирующей выполнение",
    "регламен",
    "производства техническому",
]

# ====== Лёгкая предочистка текста ======
HARD_LINEBREAKS_RE = re.compile(r'\r\n?')
SOFT_HYPHEN_RE      = re.compile('\u00AD')
NBSP_RE             = re.compile('\u00A0')
HYPHEN_BREAK_RE     = re.compile(r'(\S)-\n(\S)')
MULTI_SPACES_RE     = re.compile(r'[ \t]+')
TRIPLE_NL_RE        = re.compile(r'\n{3,}')


def clean_text(text: str) -> str:
    if not text:
        return text
    text = HARD_LINEBREAKS_RE.sub('\n', text)
    text = SOFT_HYPHEN_RE.sub('', text)
    text = NBSP_RE.sub(' ', text)
    for _ in range(2):
        text = HYPHEN_BREAK_RE.sub(r'\1\2', text)
    lines = [ln.strip() for ln in text.split('\n')]
    text = '\n'.join(lines)
    text = MULTI_SPACES_RE.sub(' ', text)
    text = TRIPLE_NL_RE.sub('\n\n', text)
    return text.strip()


def normalize_title(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def extract_meta_from_text(text: str) -> dict:
    """
    Извлекаем раздел/пункт и ключевые слова из начала чанка.
    """
    head = text[:600]
    section = None
    m = SECTION_RE.search(head)
    if m:
        roman_or_num, name = m.groups()
        name = (name or "").strip()
        section = f"Раздел {roman_or_num}" + (f" {name}" if name else "")

    clause = None
    c = CLAUSE_RE.search(head)
    if c:
        clause = c.group(1)

    kws = []
    low = text.lower()
    for k in KW_LIST:
        if k in low:
            kws.append(k)

    return {"section": section, "clause": clause, "keywords": kws}


def load_any(path: str) -> list[Document]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
        for p in pages:
            p.metadata["source"] = "pdf"
            p.metadata["title"] = os.path.basename(path)
            page_raw = p.metadata.get("page", None)
            p.metadata["page"] = (page_raw + 1) if isinstance(page_raw, int) else None
            p.page_content = clean_text(p.page_content)
        return pages

    elif ext == ".docx":
        loader = Docx2txtLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = "docx"
            d.metadata["title"] = os.path.basename(path)
            d.metadata["page"] = None
            d.page_content = clean_text(d.page_content)
        return docs

    else:
        return []

def main():
    
    for name in os.listdir(PDF_DIR):
        path = os.path.join(PDF_DIR, name)
        if not os.path.isfile(path):
            continue
        if not name.lower().endswith((".pdf", ".docx")):
            continue

        docs = load_any(path)
        raw_chunks = text_splitter.split_documents(docs)

        base_title = normalize_title(path)
        for i, chunk in enumerate(raw_chunks):
            chunk.page_content = clean_text(chunk.page_content)

            if len(chunk.page_content.strip()) < MIN_CHUNK_CHARS:
                continue


            chunk.metadata["chunk_id"] = str(uuid4())
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_start"] = chunk.metadata.get("start_index", None)

            chunk.metadata.setdefault("title", os.path.basename(path))
            chunk.metadata.setdefault("page", None)

            chunk.metadata["doc_title"] = base_title

            extra = extract_meta_from_text(chunk.page_content)
            if extra.get("section"):
                chunk.metadata["section"] = extra["section"]
            if extra.get("clause"):
                chunk.metadata["clause"] = extra["clause"]
            if extra.get("keywords"):
                kws = [k.strip() for k in extra["keywords"] if k and isinstance(k, str)]
                chunk.metadata["keywords"] = ", ".join(kws) if kws else None

            all_docs.extend([chunk])

    # --- запись в Chroma ---
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    db = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION,
    )

    print(f"\nЗагружено и сохранено {len(all_docs)} чанков.")
    if all_docs:
        print("Пример метаданных первого чанка:", all_docs[0].metadata)
