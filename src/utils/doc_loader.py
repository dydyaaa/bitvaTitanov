import os
from uuid import uuid4

import chromadb
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Настройки
CHROMA_PATH = "chroma_db/"
PDF_DIR = "docs/"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
COLLECTION = "rzd_docs"
# Эмбеддинги
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})

# Улучшенный сплиттер
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""]
)

all_docs = []
# pip install -U langchain langchain-community langchain-text-splitters langchain-huggingface langchain-chroma chromadb pypdf sentence-transformers

for file_name in os.listdir(PDF_DIR):
    if not file_name.endswith(".pdf"):
        continue

    file_path = os.path.join(PDF_DIR, file_name)
    loader = PyPDFLoader(file_path)
    pdf_pages = loader.load_and_split()

    for page in pdf_pages:
        page.metadata["source"] = "pdf"
        page.metadata["title"] = file_name
        page.metadata["page"] = page.metadata.get("page", "unknown")

    chunks: list[Document] = text_splitter.split_documents(pdf_pages)

    # Добавляем уникальные ID и индекс чанка
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(uuid4())
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_start"] = chunk.metadata.get("start_index", None)

    all_docs.extend(chunks)

client = chromadb.PersistentClient(path=CHROMA_PATH)
# Создаём БД
db = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings,
    client=client,                     
    collection_name=COLLECTION,
)


print(f"\nЗагружено и сохранено {len(all_docs)} чанков.")
