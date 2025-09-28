from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.chat.router import router as chat_router
from src.users.router import router as users_router

app = FastAPI(title="Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users_router)
app.include_router(chat_router)