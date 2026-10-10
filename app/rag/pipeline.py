from langchain_ollama import ChatOllama

from app.vectorstore.qdrant import get_vector_store
from app.vectorstore.embeddings import get_embeddings

from app.rag.memory import (
    get_history,
    add_message
)


def format_timestamp(seconds):
    seconds = int(seconds)

    minutes = seconds // 60
    seconds = seconds % 60

    return f"{minutes}:{seconds:02d}"


def get_nearby_segments(
    results,
    current_time,
    window=30
):
    nearby_segments = []

    for result in results:

        segments = result.metadata.get(
            "segments",
            []
        )

        for segment in segments:

            if (
                segment["start"]
                <= current_time + window
                and
                segment["end"]
                >= current_time - window
            ):
                nearby_segments.append(
                    segment
                )

    return nearby_segments


def find_best_segment(
    question,
    results
):

    embeddings = get_embeddings()

    question_embedding = (
        embeddings.embed_query(question)
    )

    best_segment = None
    best_score = -1

    for result in results:

        segments = result.metadata.get(
            "segments",
            []
        )

        for segment in segments:

            segment_embedding = (
                embeddings.embed_query(
                    segment["text"]
                )
            )

            dot_product = sum(
                a * b
                for a, b in zip(
                    question_embedding,
                    segment_embedding
                )
            )

            question_magnitude = (
                sum(
                    a * a
                    for a in question_embedding
                )
            ) ** 0.5

            segment_magnitude = (
                sum(
                    a * a
                    for a in segment_embedding
                )
            ) ** 0.5

            if (
                question_magnitude == 0
                or segment_magnitude == 0
            ):
                continue

            similarity = (
                dot_product
                /
                (
                    question_magnitude
                    * segment_magnitude
                )
            )

            if similarity > best_score:

                best_score = similarity
                best_segment = segment

    return best_segment


def ask_video(
    question: str,
    video_id: str,
    current_time: float
):

    # =========================================
    # 1. Connect to Qdrant
    # =========================================

    vector_store = get_vector_store()

    # =========================================
    # 2. Semantic RAG Search
    # =========================================

    semantic_results = (
        vector_store.similarity_search(
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
            }
        )
    )

    # =========================================
    # 3. Search Current Video Context
    # =========================================

    timestamp_results = (
        vector_store.similarity_search(
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
            }
        )
    )

    # =========================================
    # 4. Nearby Transcript
    # =========================================

    nearby_segments = get_nearby_segments(
        timestamp_results,
        current_time,
        window=30
    )

    nearby_context = ""

    for segment in nearby_segments:

        nearby_context += (
            f"[{format_timestamp(segment['start'])}] "
            f"{segment['text']}\n"
        )

    # =========================================
    # 5. Semantic Context
    # =========================================

    context_parts = []

    for result in semantic_results:

        context_parts.append(
            f"""
Transcript:

{result.page_content}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    # =========================================
    # 6. Conversation Memory
    # =========================================

    history = get_history(video_id)

    history_text = ""

    for message in history[-6:]:

        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    # =========================================
    # 7. No Context
    # =========================================

    if (
        not nearby_context
        and not context
    ):

        return {
            "answer": (
                "I could not find relevant "
                "information in this video."
            ),
            "timestamp": format_timestamp(
                current_time
            ),
            "youtube_url": (
                f"https://www.youtube.com/watch?v="
                f"{video_id}"
                f"&t={int(current_time)}s"
            )
        }

    # =========================================
    # 8. Ollama
    # =========================================

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    # =========================================
    # 9. Prompt
    # =========================================

    prompt = f"""
You are VideoMentor, an AI tutor.

Your job is to help the user understand the
current video clearly, like a good teacher.

Answer questions using only the provided
video transcript and conversation history.

Your job is to help the user understand the
content of the video.

Use ONLY the provided video transcript as your
source of information.

The user is currently watching the video at:

{format_timestamp(current_time)}

--------------------------------
CURRENT VIDEO CONTEXT
--------------------------------

{nearby_context}

--------------------------------
SEMANTICALLY RELEVANT CONTEXT
--------------------------------

{context}

--------------------------------
CONVERSATION HISTORY
--------------------------------

{history_text}
IMPORTANT RULES
--------------------------------

- Answer the user's question directly.
- Teach the concept instead of summarizing the video.
- Use the CURRENT VIDEO CONTEXT to understand
  what the user is referring to.
- Do not automatically describe everything happening
  in the current video context.
- Only use transcript information that is relevant
  to the question.
- Use semantic context when additional information
  is needed.
- Use conversation history to understand follow-up
  questions.
- If the user says "this", "that", "it", "here",
  or "previous", resolve the reference using the
  conversation history and current video context.
- Do not use outside knowledge.
- Do not invent information.
- Do not repeat unrelated information.
- Do not repeat the question.
- Do not say "I see you're currently at..."
  unless the timestamp itself is relevant.
- Start with the direct answer.
- Explain the concept in simple language.
- Use a small example when it helps.
- Use bullet points when they improve clarity.
- Keep normal answers concise.
- Give more detail only when the user asks for it.
- Do not mention RAG, embeddings, Qdrant, Ollama,
  context windows, or internal processing.
- If the transcript does not contain enough information,
  clearly say that the video does not provide enough
  information.
--------------------------------
USER QUESTION
--------------------------------

{question}

--------------------------------
ANSWER
--------------------------------
"""

    # =========================================
    # 10. Generate Answer
    # =========================================

    response = llm.invoke(prompt)

    answer = response.content

    # =========================================
    # 11. Save Conversation
    # =========================================

    add_message(
        video_id,
        "user",
        question
    )

    add_message(
        video_id,
        "assistant",
        answer
    )

    # =========================================
    # 12. Timestamp
    # =========================================

        # Find the most relevant transcript segment
    best_segment = find_best_segment(
        question,
        semantic_results
    )

    if best_segment:
        relevant_seconds = int(
            best_segment["start"]
        )
    else:
        relevant_seconds = int(
            current_time
        )

    timestamp = format_timestamp(
        relevant_seconds
    )

    youtube_url = (
        f"https://www.youtube.com/watch?v="
        f"{video_id}"
        f"&t={relevant_seconds}s"
    )

    return {
        "answer": answer,
        "timestamp": timestamp,
        "timestamp_seconds": relevant_seconds,
        "youtube_url": youtube_url
    }

    # =========================================
    # 13. Return
    # =========================================

    return {
        "answer": answer,
        "timestamp": timestamp,
        "youtube_url": youtube_url
    }