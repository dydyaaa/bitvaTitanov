import os

from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.docs.models import Document

DOCS_DIR = "src/docs/files"
os.makedirs(DOCS_DIR, exist_ok=True)


async def save_file(file: UploadFile, db: AsyncSession) -> Document:

    if file.content_type not in [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    ]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    file_path = os.path.join(DOCS_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    doc = Document(
        filename=file.filename,
        original_name=file.filename,
        content_type=file.content_type,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    return doc


async def get_document(doc_id: int, db: AsyncSession) -> Document:
    """Получить документ по ID"""
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalars().first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


async def delete_document(doc: Document, db: AsyncSession) -> None:
    """Удалить документ из БД и с диска"""
    file_path = os.path.join(DOCS_DIR, doc.filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    await db.delete(doc)
    await db.commit()


async def list_documents(db: AsyncSession) -> list[Document]:
    """Получить список всех документов"""
    result = await db.execute(select(Document))
    return result.scalars().all()
