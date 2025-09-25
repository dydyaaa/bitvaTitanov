from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from src.chat.models import Message


async def add_message(db: AsyncSession, user_id: int, text: str):
    bot_response = "Привет" # Сюда метод ответа или че нибудь такое, хз
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

