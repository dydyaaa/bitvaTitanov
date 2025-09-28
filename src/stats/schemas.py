from datetime import datetime
from typing import List

from pydantic import BaseModel


class UserStatsResponse(BaseModel):
    period: str
    date: datetime
    unique_users: int


class WordFrequency(BaseModel):
    word: str
    frequency: int


class TopWordsResponse(BaseModel):
    top_words: List[WordFrequency]
    total_words: int


class StatsPeriod(BaseModel):
    period: str  # "day", "week", "month"
