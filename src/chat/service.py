import time

import torch
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.chat.models import Message
from src.utils.rag_service import rag


async def add_message(db: AsyncSession, user_id: int, text: str):
    t0 = time.perf_counter()
    # если есть CUDA — синхронизируем для честного старта
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    bot_response = rag.generate_answer(user_id=user_id, query=text)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"[RAG] total latency: {dt:.3f} s")

    msg = Message(user_id=user_id, text=text, response=bot_response)
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg

async def get_history(db: AsyncSession, user_id: int):
    query = select(Message).where(Message.user_id == user_id).order_by(Message.created_at.desc())
    result = await db.execute(query)
    return result.scalars().all()

async def clear_history(db: AsyncSession, user_id: int):
    stmt = delete(Message).where(Message.user_id == user_id)
    await db.execute(stmt)
    await db.commit()

