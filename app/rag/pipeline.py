from langchain_ollama import ChatOllama

from app.vectorstore.qdrant import get_vector_store
from app.vectorstore.embeddings import get_embeddings

from app.rag.memory import (
    get_history,
    add_message,
)


# =========================================
# Tutor Mode Instructions
# =========================================

TUTOR_MODES = {
    "explain": """
You are in EXPLAIN mode.

Your job is to teach the requested concept clearly.

- Explain concepts in beginner-friendly language.
- Break complicated ideas into smaller steps.
- Use examples when helpful.
- Explain technical terms simply.
- Focus on understanding rather than memorization.
""",

    "summarize": """
You are in SUMMARIZE mode.

Your job is to summarize the relevant video content.

- Identify the main ideas.
- Present important points as bullet points.
- Preserve important technical details.
- Avoid unnecessary repetition.
- Include timestamps when useful.
- Do not add information absent from the transcript.
""",

    "quiz": """
You are in QUIZ mode.

Your job is to help the user test their understanding.

- Generate questions based on the video transcript.
- Prefer one question at a time unless multiple questions
  are explicitly requested.
- Start with a clear question.
- Do not immediately reveal the answer.
- When the user answers, evaluate their response.
- Explain what they got right and what they missed.
- Provide the correct answer when appropriate.
""",

    "interview": """
You are in INTERVIEW mode.

Your job is to act as a technical interviewer.

- Ask interview-style questions based on the video.
- Prefer one question at a time.
- Wait for the user's response.
- Evaluate correctness, clarity, and completeness.
- Give constructive feedback.
- Explain the ideal answer when appropriate.
- Increase difficulty gradually when possible.
""",
}


def format_timestamp(seconds):
    """Convert seconds into MM:SS format."""

    seconds = int(seconds or 0)

    minutes = seconds // 60
    seconds = seconds % 60

    return f"{minutes}:{seconds:02d}"


def get_nearby_segments(results, current_time, window=30):
    """Find transcript segments near the current timestamp."""

    nearby_segments = []

    for result in results:
        segments = result.metadata.get("segments", [])

        for segment in segments:
            if (
                segment["start"] <= current_time + window
                and segment["end"] >= current_time - window
            ):
                nearby_segments.append(segment)

    return nearby_segments


def find_best_segment(question, results):
    """Find the transcript segment most similar to the question."""

    embeddings = get_embeddings()

    question_embedding = embeddings.embed_query(question)

    best_segment = None
    best_score = -1

    question_magnitude = (
        sum(value * value for value in question_embedding) ** 0.5
    )

    if question_magnitude == 0:
        return None

    for result in results:
        segments = result.metadata.get("segments", [])

        for segment in segments:
            segment_embedding = embeddings.embed_query(
                segment["text"]
            )

            segment_magnitude = (
                sum(value * value for value in segment_embedding) ** 0.5
            )

            if segment_magnitude == 0:
                continue

            dot_product = sum(
                a * b
                for a, b in zip(
                    question_embedding,
                    segment_embedding,
                )
            )

            similarity = (
                dot_product
                / (question_magnitude * segment_magnitude)
            )

            if similarity > best_score:
                best_score = similarity
                best_segment = segment

    return best_segment


def ask_video(
    question: str,
    video_id: str,
    current_time: float,
    conversation_id: str,
    mode: str = "explain",
):
    """
    Answer questions about a YouTube video using RAG.

    Supported modes:
    explain, summarize, quiz, interview
    """

    # =========================================
    # 1. Validate Tutor Mode
    # =========================================

    mode = (mode or "explain").lower().strip()

    if mode not in TUTOR_MODES:
        mode = "explain"

    mode_instructions = TUTOR_MODES[mode]

    # =========================================
    # 2. Connect to Qdrant
    # =========================================

    vector_store = get_vector_store()

    # =========================================
    # 3. Semantic RAG Search
    # =========================================

    semantic_results = vector_store.similarity_search(
        question,
        k=3,
        filter={
            "must": [
                {
                    "key": "metadata.video_id",
                    "match": {
                        "value": video_id
                    }
                }
            ]
        },
    )

    # =========================================
    # 4. Retrieve Current Video Context
    # =========================================

    timestamp_results = vector_store.similarity_search(
        "video transcript",
        k=10,
        filter={
            "must": [
                {
                    "key": "metadata.video_id",
                    "match": {
                        "value": video_id
                    }
                }
            ]
        },
    )

    # =========================================
    # 5. Retrieve Nearby Transcript Segments
    # =========================================

    nearby_segments = get_nearby_segments(
        timestamp_results,
        current_time,
        window=30,
    )

    nearby_context = ""

    for segment in nearby_segments:
        nearby_context += (
            f"[{format_timestamp(segment['start'])}] "
            f"{segment['text']}\n"
        )

    # =========================================
    # 6. Build Semantic Context
    # =========================================

    context_parts = []

    for result in semantic_results:
        context_parts.append(
            f"Transcript:\n{result.page_content}"
        )

    context = "\n\n".join(context_parts)

    # =========================================
    # 7. Retrieve Conversation History
    # =========================================

    history = get_history(conversation_id)

    history_text = ""

    for message in history[-10:]:
        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    # =========================================
    # 8. Handle Missing Context
    # =========================================

    if not nearby_context.strip() and not context.strip():

        answer = (
            "I could not find relevant information in this video. "
            "Try asking about a topic discussed in the transcript."
        )

        add_message(
            conversation_id,
            "user",
            question,
            video_id=video_id,
        )

        add_message(
            conversation_id,
            "assistant",
            answer,
            video_id=video_id,
        )

        return {
            "answer": answer,
            "timestamp": format_timestamp(current_time),
            "timestamp_seconds": int(current_time),
            "youtube_url": (
                f"https://www.youtube.com/watch?v="
                f"{video_id}&t={int(current_time)}s"
            ),
            "mode": mode,
        }

    # =========================================
    # 9. Initialize Ollama
    # =========================================

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
    )

    # =========================================
    # 10. Build Tutor Prompt
    # =========================================

    prompt = f"""
You are VideoMentor, an AI tutor helping the user learn
from the current YouTube video.

CURRENT TUTOR MODE
--------------------------------

{mode_instructions}

CURRENT VIDEO TIMESTAMP
--------------------------------

{format_timestamp(current_time)}

CURRENT VIDEO CONTEXT
--------------------------------

{nearby_context}

SEMANTICALLY RELEVANT TRANSCRIPT
--------------------------------

{context}

CONVERSATION HISTORY
--------------------------------

{history_text}

GENERAL RULES
--------------------------------

- Ground factual claims about the video in the supplied transcript.
- Do not invent information about what the video says.
- Use conversation history to understand follow-up questions.
- Resolve references such as "this", "that", "it", and "previous"
  using the conversation history and current video context.
- If the transcript does not contain enough information,
  clearly state that limitation.
- Do not mention RAG, embeddings, Qdrant, Ollama, or internal
  processing.
- Follow the selected tutor mode.
- Keep responses clear and useful.

USER MESSAGE
--------------------------------

{question}

RESPONSE
--------------------------------
"""

    # =========================================
    # 11. Generate Answer
    # =========================================

    response = llm.invoke(prompt)

    answer = response.content

    # =========================================
    # 12. Save Conversation
    # =========================================

    add_message(
        conversation_id,
        "user",
        question,
        video_id=video_id,
    )

    add_message(
        conversation_id,
        "assistant",
        answer,
        video_id=video_id,
    )

    # =========================================
    # 13. Find Relevant Timestamp
    # =========================================

    best_segment = find_best_segment(
        question,
        semantic_results,
    )

    if best_segment:
        relevant_seconds = int(best_segment["start"])
    else:
        relevant_seconds = int(current_time)

    timestamp = format_timestamp(relevant_seconds)

    youtube_url = (
        f"https://www.youtube.com/watch?v="
        f"{video_id}&t={relevant_seconds}s"
    )

    # =========================================
    # 14. Return Response
    # =========================================

    return {
        "answer": answer,
        "timestamp": timestamp,
        "timestamp_seconds": relevant_seconds,
        "youtube_url": youtube_url,
        "mode": mode,
    }