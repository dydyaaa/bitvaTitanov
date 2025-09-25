from datetime import datetime

from pydantic import BaseModel


class MessageCreate(BaseModel):
    text: str

class MessageOut(BaseModel):
    id: int
    text: str
    response: str
    created_at: datetime

    class Config:
        orm_mode = True
