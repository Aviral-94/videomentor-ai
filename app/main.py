from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag.pipeline import ask_video


app = FastAPI(
    title="VideoMentor AI",
    description="AI-powered YouTube learning assistant",
    version="1.0.0"
)


# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    video_id: str
    current_time: float
    question: str


@app.get("/")
def root():
    return {
        "message": "VideoMentor AI API is running"
    }


@app.post("/api/chat")
def chat(request: ChatRequest):

    result = ask_video(
        request.question,
        request.video_id,
        request.current_time
    )

    return result