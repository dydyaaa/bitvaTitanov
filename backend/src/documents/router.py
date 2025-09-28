from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.documents import service
from src.documents.schemas import DocumentBase, DocumentList

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentBase)
async def upload_file(
    file: Annotated[UploadFile, File()], 
    db: AsyncSession = Depends(get_db)
):
    """Загрузить файл в папку src/docs"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не выбран")
    
    doc = await service.save_file(file, db)
    return DocumentBase.model_validate(doc, from_attributes=True)


@router.get("/list", response_model=DocumentList)
async def list_files(
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
    db: AsyncSession = Depends(get_db)
):
    """Просмотреть список файлов с пагинацией"""
    documents, total = await service.list_documents(db, skip=skip, limit=limit)
    
    return DocumentList(
        documents=[DocumentBase.model_validate(doc, from_attributes=True) for doc in documents],
        total=total
    )




@router.delete("/{doc_id}")
async def delete_file(doc_id: int, db: AsyncSession = Depends(get_db)):
    """Удалить файл по ID"""
    doc = await service.get_document(doc_id, db)
    await service.delete_document(doc, db)
    return {"status": "deleted", "message": f"Файл '{doc.original_name}' успешно удален"}
