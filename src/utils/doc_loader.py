# import os
# from uuid import uuid4

# import chromadb
# from langchain.schema.document import Document
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # Настройки
# CHROMA_PATH = "src/chroma_db/"
# PDF_DIR = "src/docs/"
# EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
# COLLECTION = "rzd_docs"
# # Эмбеддинги
# embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})

# # Улучшенный сплиттер
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     add_start_index=True,
#     separators=["\n\n", "\n", ".", " ", ""]
# )

# all_docs = []
# # pip install -U langchain langchain-community langchain-text-splitters langchain-huggingface langchain-chroma chromadb pypdf sentence-transformers

# for file_name in os.listdir(PDF_DIR):
#     if not file_name.endswith(".pdf"):
#         continue

#     file_path = os.path.join(PDF_DIR, file_name)
#     loader = PyPDFLoader(file_path)
#     pdf_pages = loader.load_and_split()

#     for page in pdf_pages:
#         page.metadata["source"] = "pdf"
#         page.metadata["title"] = file_name
#         page.metadata["page"] = page.metadata.get("page", "unknown")

#     chunks: list[Document] = text_splitter.split_documents(pdf_pages)

#     # Добавляем уникальные ID и индекс чанка
#     for i, chunk in enumerate(chunks):
#         chunk.metadata["chunk_id"] = str(uuid4())
#         chunk.metadata["chunk_index"] = i
#         chunk.metadata["chunk_start"] = chunk.metadata.get("start_index", None)

#     all_docs.extend(chunks)

# client = chromadb.PersistentClient(path=CHROMA_PATH)
# # Создаём БД
# db = Chroma.from_documents(
#     documents=all_docs,
#     embedding=embeddings,
#     client=client,                     
#     collection_name=COLLECTION,
# )


# print(f"\nЗагружено и сохранено {len(all_docs)} чанков.")
import os
from uuid import uuid4

import chromadb
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

CHROMA_PATH = "src/chroma_db/"
PDF_DIR = "src/docs/"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
COLLECTION = "rzd_docs"

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

def load_any(path: str) -> list[Document]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        # постраничная загрузка
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
        # нормализуем метаданные страниц
        for p in pages:
            p.metadata["source"] = "pdf"
            p.metadata["title"] = os.path.basename(path)
            p.metadata["page"] = p.metadata.get("page", "unknown")
        return pages

    elif ext == ".docx":
        # TXT-извлечение всего документа (без «страниц»)
        loader = Docx2txtLoader(path)
        docs = loader.load()  # обычно один Document с целым текстом
        for d in docs:
            d.metadata["source"] = "docx"
            d.metadata["title"] = os.path.basename(path)
            d.metadata["page"] = None   # у docx нет страниц
        return docs

    else:
        return []

# --- основной цикл по файлам ---
for name in os.listdir(PDF_DIR):
    path = os.path.join(PDF_DIR, name)
    if not os.path.isfile(path):
        continue
    if not name.lower().endswith((".pdf", ".docx")):
        continue

    docs = load_any(path)

    # режем на чанки одинаковым сплиттером
    chunks = text_splitter.split_documents(docs)

    # добавляем служебные метаданные
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(uuid4())
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_start"] = chunk.metadata.get("start_index", None)
        # гарантируем наличие title/page
        chunk.metadata.setdefault("title", os.path.basename(path))
        chunk.metadata.setdefault("page", None)

    all_docs.extend(chunks)

# --- запись в Chroma ---
client = chromadb.PersistentClient(path=CHROMA_PATH)
db = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings,
    client=client,
    collection_name=COLLECTION,
)

print(f"\nЗагружено и сохранено {len(all_docs)} чанков.")
