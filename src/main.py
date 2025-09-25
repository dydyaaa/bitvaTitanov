from fastapi import FastAPI
from src.users.router import router as users_router
from src.chat.router import router as chat_router

app = FastAPI(title="Chat API")

app.include_router(users_router)
app.include_router(chat_router)
