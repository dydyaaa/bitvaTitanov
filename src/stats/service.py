import re
from collections import Counter
from datetime import datetime, timedelta
from typing import List

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.chat.models import Message
from src.stats.schemas import TopWordsResponse, UserStatsResponse, WordFrequency


async def get_unique_users_count(
    db: AsyncSession, period: str
) -> UserStatsResponse:
    """Получить количество уникальных пользователей за период"""
    
    now = datetime.now()
    
    if period == "day":
        start_date = now - timedelta(days=1)
    elif period == "week":
        start_date = now - timedelta(weeks=1)
    elif period == "month":
        start_date = now - timedelta(days=30)
    else:
        raise ValueError("Period must be 'day', 'week', or 'month'")
    
    # Запрос для подсчета уникальных пользователей за период
    query = select(func.count(func.distinct(Message.user_id))).where(
        Message.created_at >= start_date
    )
    
    result = await db.execute(query)
    unique_users_count = result.scalar() or 0
    
    return UserStatsResponse(
        period=period,
        date=now,
        unique_users=unique_users_count
    )


def clean_text(text: str) -> str:
    """Очистить текст от специальных символов"""
    # Удаляем все символы кроме букв, цифр и пробелов
    cleaned = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    # Удаляем лишние пробелы
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def filter_words(words: List[str]) -> List[str]:
    """Фильтровать слова: убрать короткие слова (менее 5 символов)"""
    return [word for word in words if len(word) >= 5]


async def get_top_words(db: AsyncSession) -> TopWordsResponse:
    """Получить топ слов из всех вопросов пользователей"""
    
    # Получаем все тексты сообщений пользователей (не удаленных)
    query = select(Message.text).where(
        Message.text.isnot(None)
    )
    
    result = await db.execute(query)
    messages = result.scalars().all()
    
    # Объединяем все тексты в один
    all_text = " ".join(messages).lower()
    
    # Очищаем от специальных символов
    cleaned_text = clean_text(all_text)
    
    # Разбиваем на слова
    words = cleaned_text.split()
    
    # Фильтруем короткие слова
    filtered_words = filter_words(words)
    
    # Подсчитываем частоту
    word_counter = Counter(filtered_words)
    
    # Получаем топ 10 слов
    top_words = word_counter.most_common(10)
    
    # Формируем ответ
    word_frequencies = [
        WordFrequency(word=word, frequency=freq) 
        for word, freq in top_words
    ]
    
    return TopWordsResponse(
        top_words=word_frequencies,
        total_words=len(filtered_words)
    )
