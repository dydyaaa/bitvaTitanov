from fastapi import FastAPI

from src.chat.router import router as chat_router
from src.users.router import router as users_router
from src.docs.router import router as docs_router

app = FastAPI(title="Chat API")

app.include_router(users_router)
app.include_router(chat_router)
app.include_router(docs_router)