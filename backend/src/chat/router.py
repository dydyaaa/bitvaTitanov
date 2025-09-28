from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.chat import schemas, service
from src.core.database import get_db
from src.core.security import decode_token

router = APIRouter(prefix="/chat", tags=["chat"])

async def get_current_user(
    authorization: str = Header(...),
):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ")[1]
    payload = decode_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return int(payload["sub"])


@router.post("/send", response_model=schemas.MessageOut)
async def send_message(
    msg: schemas.MessageCreate,
    user_id: int = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.add_message(db, user_id, msg.text)


@router.get("/history")
async def get_history(
    user_id: int = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.get_history(db, user_id)


@router.delete("/history")
async def clear_history(
    user_id: int = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await service.clear_history(db, user_id)
    return {"status": "cleared"}
