from datetime import datetime

from pydantic import BaseModel


class DocumentBase(BaseModel):
    id: int
    filename: str
    original_name: str
    content_type: str
    file_size: int | None
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentCreate(BaseModel):
    original_name: str
    content_type: str
    file_size: int | None = None


class DocumentList(BaseModel):
    documents: list[DocumentBase]
    total: int
