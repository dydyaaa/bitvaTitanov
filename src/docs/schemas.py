from datetime import datetime

from pydantic import BaseModel


class DocumentBase(BaseModel):
    id: int
    filename: str
    original_name: str
    content_type: str
    created_at: datetime

    class Config:
        orm_mode = True
