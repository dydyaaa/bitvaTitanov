from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.stats import service
from src.stats.schemas import StatsPeriod, TopWordsResponse, UserStatsResponse

router = APIRouter(prefix="/stats", tags=["statistics"])


@router.post("/users", response_model=UserStatsResponse)
async def get_unique_users_count(
    period_data: StatsPeriod,
    db: AsyncSession = Depends(get_db)
):
    """Получить количество уникальных пользователей за период"""
    
    if period_data.period not in ["day", "week", "month"]:
        raise HTTPException(
            status_code=400, 
            detail="Period must be 'day', 'week', or 'month'"
        )
    
    return await service.get_unique_users_count(db, period_data.period)


@router.get("/top-words", response_model=TopWordsResponse)
async def get_top_words(
    db: AsyncSession = Depends(get_db)
):
    """Получить топ 10 слов из всех вопросов пользователей
    
    Обрабатывает все тексты сообщений:
    - Удаляет специальные символы
    - Удаляет слова короче 5 символов
    - Возвращает топ 10 слов с частотой использования
    """
    
    return await service.get_top_words(db)
