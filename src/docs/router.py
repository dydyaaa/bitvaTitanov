from fastapi import APIRouter, UploadFile, Depends
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.docs.schemas import DocumentBase
from src.docs import service

router = APIRouter(prefix="/docs", tags=["documents"])


@router.post("/upload", response_model=DocumentBase)
async def upload_file(file: UploadFile, db: AsyncSession = Depends(get_db)):
    return await service.save_file(file, db)


@router.get("/{doc_id}", response_model=DocumentBase)
async def get_doc_info(doc_id: int, db: AsyncSession = Depends(get_db)):
    return await service.get_document(doc_id, db)


@router.get("/{doc_id}/download")
async def download_file(doc_id: int, db: AsyncSession = Depends(get_db)):
    doc = await service.get_document(doc_id, db)
    return FileResponse(
        path=f"src/docs/files/{doc.filename}",
        filename=doc.original_name,
        media_type=doc.content_type,
    )


@router.delete("/{doc_id}")
async def delete_file(doc_id: int, db: AsyncSession = Depends(get_db)):
    doc = await service.get_document(doc_id, db)
    await service.delete_document(doc, db)
    return {"status": "deleted"}


@router.get("/list", response_model=list[DocumentBase])
async def list_files(db: AsyncSession = Depends(get_db)):
    return await service.list_documents(db)
