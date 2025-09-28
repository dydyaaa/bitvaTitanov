import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.documents.models import Document

DOCS_DIR = Path("src/docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)


async def save_file(file: UploadFile, db: AsyncSession) -> Document:
    """Сохранить загруженный файл"""
    
    # Проверяем поддерживаемые типы файлов
    allowed_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
        "application/msword",  # .doc
        "text/plain"
    ]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Неподдерживаемый тип файла: {file.content_type}"
        )

    # Генерируем уникальное имя файла
    file_extension = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = DOCS_DIR / unique_filename

    # Сохраняем файл
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            file_size = len(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении файла: {str(e)}")

    # Сохраняем информацию в БД
    doc = Document(
        filename=unique_filename,
        original_name=file.filename,
        content_type=file.content_type,
        file_size=file_size,
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
        raise HTTPException(status_code=404, detail="Документ не найден")
    return doc


async def delete_document(doc: Document, db: AsyncSession) -> None:
    """Удалить документ из БД и с диска"""
    file_path = DOCS_DIR / doc.filename
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при удалении файла: {str(e)}")

    await db.delete(doc)
    await db.commit()


async def list_documents(db: AsyncSession, skip: int = 0, limit: int = 100) -> tuple[list[Document], int]:
    """Получить список всех документов с пагинацией"""
    # Получаем общее количество
    count_result = await db.execute(select(Document))
    total = len(count_result.scalars().all())
    
    # Получаем документы с пагинацией
    result = await db.execute(
        select(Document)
        .offset(skip)
        .limit(limit)
        .order_by(Document.created_at.desc())
    )
    documents = result.scalars().all()
    
    return documents, total


