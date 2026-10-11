from langchain_ollama import ChatOllama

from app.vectorstore.qdrant import get_vector_store


def generate_video_notes(video_id: str) -> str:
    """
    Generate structured study notes from an indexed YouTube video.
    Uses the existing Qdrant vector store and local Ollama model.
    """

    video_id = video_id.strip()

    if not video_id:
        raise ValueError("A valid video ID is required.")

    # 1. Connect to the existing vector store.
    vector_store = get_vector_store()

    # 2. Retrieve transcript chunks belonging to this video.
    results = vector_store.similarity_search(
        "main topics concepts explanations definitions examples",
        k=10,
        filter={
            "must": [
                {
                    "key": "metadata.video_id",
                    "match": {
                        "value": video_id,
                    },
                }
            ]
        },
    )

    # 3. Check whether transcript content was found.
    if not results:
        raise ValueError(
            "No indexed transcript was found for this video. "
            "Please index the video before generating notes."
        )

    # 4. Build transcript context.
    transcript_parts = []

    for result in results:
        text = result.page_content.strip()

        if not text:
            continue

        segments = result.metadata.get("segments", [])

        if segments:
            for segment in segments:
                start = int(segment.get("start", 0))
                minutes = start // 60
                seconds = start % 60

                transcript_parts.append(
                    f"[{minutes}:{seconds:02d}] "
                    f"{segment.get('text', '').strip()}"
                )
        else:
            transcript_parts.append(text)

    # Remove duplicate chunks while preserving order.
    transcript_parts = list(dict.fromkeys(transcript_parts))

    transcript = "\n\n".join(transcript_parts)

    if not transcript.strip():
        raise ValueError(
            "The indexed video does not contain usable transcript text."
        )

    # 5. Initialize the existing local LLM.
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
    )

    # 6. Generate structured study notes.
    prompt = f"""
You are VideoMentor AI, an educational note-taking assistant.

Create clear, useful study notes using ONLY the supplied
YouTube transcript.

TRANSCRIPT
----------------------------

{transcript}

NOTE-GENERATION INSTRUCTIONS
----------------------------

Organize the notes using these sections:

# Study Notes

## 1. Overview
Briefly explain the main topic of the video.

## 2. Key Concepts
Explain the important concepts in beginner-friendly language.

## 3. Important Definitions
List important terms and explain their meanings.

## 4. Detailed Explanation
Explain the major ideas step by step.
Use headings and bullet points where appropriate.

## 5. Examples
Include examples mentioned in the transcript.
Do not invent examples and claim they came from the video.

## 6. Important Timestamps
List relevant timestamps when the transcript provides them.
Use MM:SS format.

## 7. Key Takeaways
List the most important things a student should remember.

RULES
----------------------------

- Use only information supported by the transcript.
- Do not invent facts, definitions, or timestamps.
- If a section is not supported by the transcript, say so briefly.
- Explain technical concepts in simple language.
- Avoid unnecessary repetition.
- Make the notes useful for revision.
- Do not mention Qdrant, embeddings, RAG, or Ollama.
- Return the notes as readable Markdown.

Generate the study notes now.
"""

    response = llm.invoke(prompt)

    notes = response.content

    if not notes or not notes.strip():
        raise RuntimeError("The AI returned empty study notes.")

    return notes.strip()