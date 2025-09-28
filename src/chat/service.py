import time

import torch
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.chat.models import Message


async def add_message(db: AsyncSession, user_id: int, text: str):

    bot_response = "test"
    msg = Message(user_id=user_id, text=text, response=bot_response)
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg

async def get_history(db: AsyncSession, user_id: int):
    query = select(Message).where(
        Message.user_id == user_id,
        Message.is_deleted == False
        ).order_by(Message.created_at.desc())
    result = await db.execute(query)
    return result.scalars().all()

async def clear_history(db: AsyncSession, user_id: int):
    stmt = update(Message).where(
        Message.user_id == user_id,
        Message.is_deleted == False
        ).values(is_deleted=True)
    await db.execute(stmt)
    await db.commit()

