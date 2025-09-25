from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.chat.models import Message


async def add_message(db: AsyncSession, user_id: int, text: str):
    bot_response = "Привет" # Сюда метод ответа или че нибудь такое, хз
    msg = Message(user_id=user_id, text=text, response=bot_response)
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg

async def get_history(db: AsyncSession, user_id: int):
    result = await db.execute(
        text("SELECT * FROM messages WHERE user_id = :user_id ORDER BY created_at DESC"),
        {"user_id": user_id}
    )
    return result.fetchall()

async def clear_history(db: AsyncSession, user_id: int):
    await db.execute(f"DELETE FROM messages WHERE user_id = {user_id}")
    await db.commit()
