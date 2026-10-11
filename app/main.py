from fastapi import HTTPException
from app.rag.notes import generate_video_notes
from app.rag.quiz import generate_quiz
from app.rag.study_guide import generate_study_guide
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag.pipeline import ask_video
from app.rag.memory import get_history, clear_memory


app = FastAPI(
    title="VideoMentor AI",
    description="AI-powered YouTube learning assistant",
    version="1.2.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    conversation_id: str
    video_id: str
    current_time: float
    question: str
    mode: str = "explain"

class NotesRequest(BaseModel):
    video_id: str
@app.get("/")
def root():
    return {
        "message": "VideoMentor AI API is running"
    }
class StudyGuideRequest(BaseModel):
    video_id: str
class QuizRequest(BaseModel):
    video_id: str
    num_questions: int = 5
    difficulty: str = "medium"
@app.post("/api/chat")
def chat(request: ChatRequest):

    result = ask_video(
        question=request.question,
        video_id=request.video_id,
        current_time=request.current_time,
        conversation_id=request.conversation_id,
        mode=request.mode,
    )

    return {
        **result,
        "conversation_id": request.conversation_id,
    }


@app.get("/api/chat/history/{conversation_id}")
def chat_history(conversation_id: str):

    return {
        "conversation_id": conversation_id,
        "messages": get_history(conversation_id),
    }


@app.delete("/api/chat/history/{conversation_id}")
def delete_chat_history(conversation_id: str):

    clear_memory(conversation_id)

    return {
        "conversation_id": conversation_id,
        "message": "Conversation memory cleared",
    }
@app.post("/api/notes")
def generate_notes(request: NotesRequest):
    try:
        notes = generate_video_notes(request.video_id)

        return {
            "video_id": request.video_id,
            "notes": notes,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:
        print(f"Notes generation failed: {error}")

        raise HTTPException(
            status_code=500,
            detail="Failed to generate notes. Please try again.",
        )
@app.post("/api/study-guide")
def create_study_guide(request: StudyGuideRequest):
    try:
        return generate_study_guide(request.video_id)

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:
        print(f"Study guide generation failed: {error}")

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to generate the study guide. "
                "Check the backend terminal for details."
            ),
        )
@app.post("/api/quiz")
def create_quiz(request: QuizRequest):
    try:
        return generate_quiz(
            video_id=request.video_id,
            num_questions=request.num_questions,
            difficulty=request.difficulty,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        print(f"Quiz generation failed: {error}")

        raise HTTPException(
            status_code=500,
            detail="Failed to generate the quiz. Check the backend terminal.",
        )