import json
import os
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag.pipeline import ask_video
from app.rag.memory import get_history, clear_memory
from app.rag.notes import generate_video_notes
from app.rag.quiz import generate_quiz
from app.rag.study_guide import generate_study_guide
from app.ingestion.youtube import extract_video_id, fetch_transcript
from app.ingestion.chunker import create_chunks
from app.vectorstore.qdrant import store_documents


app = FastAPI(
    title="VideoMentor AI",
    description="AI-powered YouTube learning assistant",
    version="1.4.0",
)

# Dynamic CORS origins from environment variable, preserving local Vite dev server fallback
DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
env_cors = os.getenv("CORS_ORIGINS", "")
if env_cors.strip():
    allowed_origins = [origin.strip() for origin in env_cors.split(",") if origin.strip()]
    for origin in DEFAULT_CORS_ORIGINS:
        if origin not in allowed_origins:
            allowed_origins.append(origin)
else:
    allowed_origins = DEFAULT_CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent registry for videos ingested through this API.
# It intentionally does not modify the existing Qdrant collection or qdrant.py.
REGISTRY_PATH = Path(__file__).resolve().parent / "video_registry.json"
_registry_lock = Lock()


def read_registry():
    with _registry_lock:
        if not REGISTRY_PATH.exists():
            return {}
        try:
            data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}


def write_registry(registry):
    with _registry_lock:
        REGISTRY_PATH.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


class ChatRequest(BaseModel):
    conversation_id: str
    video_id: str
    video_ids: list[str] | None = None
    current_time: float = 0
    question: str
    mode: str = "explain"


class NotesRequest(BaseModel):
    video_id: str


class StudyGuideRequest(BaseModel):
    video_id: str


class QuizRequest(BaseModel):
    video_id: str
    num_questions: int = 5
    difficulty: str = "medium"


class IngestVideoRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2048)


@app.get("/")
def root():
    return {
        "message": "VideoMentor AI API is running",
        "version": "1.4.0",
    }


@app.post("/api/videos/ingest")
def ingest_video(request: IngestVideoRequest):
    try:
        video_id = extract_video_id(request.url.strip())
    except (ValueError, IndexError) as error:
        raise HTTPException(status_code=400, detail="Please provide a valid YouTube video URL.") from error

    registry = read_registry()
    if video_id in registry and registry[video_id].get("status") == "indexed":
        return {
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "status": "already_indexed",
            "chunks": registry[video_id].get("chunks", 0),
            "message": "This video is already in the VideoMentor library.",
        }

    try:
        entries = fetch_transcript(video_id)
        if not entries:
            raise ValueError("No transcript entries were available for this video.")

        chunks = create_chunks(entries)
        if not chunks:
            raise ValueError("Could not create searchable chunks from this transcript.")

        store_documents(chunks)

        registry[video_id] = {
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "status": "indexed",
            "chunks": len(chunks),
        }
        write_registry(registry)

        return {
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "status": "indexed",
            "transcript_entries": len(entries),
            "chunks": len(chunks),
            "message": "Video transcript successfully indexed.",
        }

    except HTTPException:
        raise
    except Exception as error:
        print(f"Video ingestion failed for {video_id}: {error}")
        message = str(error).strip() or "Transcript ingestion failed."
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not ingest this video. It may not have an accessible transcript, "
                "or YouTube may be temporarily blocking transcript requests. "
                f"Details: {message}"
            ),
        ) from error


@app.get("/api/videos")
def list_videos():
    registry = read_registry()
    return {"videos": list(registry.values())}


@app.delete("/api/videos/{video_id}")
def remove_video_from_library(video_id: str):
    # This only removes the local library entry. It does not delete Qdrant vectors.
    registry = read_registry()
    existed = registry.pop(video_id, None) is not None
    write_registry(registry)
    return {
        "video_id": video_id,
        "removed": existed,
        "message": (
            "Removed from the saved library. Existing Qdrant vectors were not deleted."
            if existed else "Video was not in the saved library."
        ),
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        result = ask_video(
            question=request.question,
            video_id=request.video_id,
            current_time=request.current_time,
            conversation_id=request.conversation_id,
            mode=request.mode,
            video_ids=request.video_ids,
        )
        return {**result, "conversation_id": request.conversation_id}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        print(f"Chat failed: {error}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer. Check the backend terminal.",
        ) from error


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
        return {"video_id": request.video_id, "notes": generate_video_notes(request.video_id)}
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        print(f"Notes generation failed: {error}")
        raise HTTPException(status_code=500, detail="Failed to generate notes. Please try again.") from error


@app.post("/api/study-guide")
def create_study_guide(request: StudyGuideRequest):
    try:
        return generate_study_guide(request.video_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        print(f"Study guide generation failed: {error}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate the study guide. Check the backend terminal for details.",
        ) from error


@app.post("/api/quiz")
def create_quiz(request: QuizRequest):
    try:
        return generate_quiz(
            video_id=request.video_id,
            num_questions=request.num_questions,
            difficulty=request.difficulty,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        print(f"Quiz generation failed: {error}")
        raise HTTPException(status_code=500, detail="Failed to generate the quiz. Check the backend terminal.") from error
