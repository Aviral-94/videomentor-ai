from langchain_ollama import ChatOllama

from app.vectorstore.qdrant import get_vector_store


def generate_study_guide(video_id: str) -> dict:
    """
    Generate structured study notes and a Mermaid flowchart
    from an indexed YouTube video's transcript.

    Does not modify the existing Qdrant implementation.
    """

    video_id = video_id.strip()

    if not video_id:
        raise ValueError("A valid video ID is required.")

    vector_store = get_vector_store()

    results = vector_store.similarity_search(
        "main topics, important concepts, definitions, explanations, "
        "examples, processes and relationships between concepts",
        k=10,
        filter={
            "must": [
                {
                    "key": "metadata.video_id",
                    "match": {
                        "value": video_id
                    },
                }
            ]
        },
    )

    if not results:
        raise ValueError(
            "No indexed transcript was found for this video. "
            "Please index the video first."
        )

    # Collect transcript text and timestamped segments.
    transcript_parts = []
    seen_segments = set()

    for result in results:
        metadata = result.metadata or {}
        segments = metadata.get("segments", [])

        if segments:
            for segment in segments:
                text = segment.get("text", "").strip()

                if not text:
                    continue

                start = segment.get("start", 0)

                try:
                    start = int(float(start))
                except (TypeError, ValueError):
                    start = 0

                if start < 0:
                    start = 0

                minutes = start // 60
                seconds = start % 60

                key = (start, text)

                if key in seen_segments:
                    continue

                seen_segments.add(key)

                transcript_parts.append(
                    {
                        "timestamp_seconds": start,
                        "timestamp": f"{minutes}:{seconds:02d}",
                        "text": text,
                    }
                )

        else:
            # Preserve usable chunks even when segment metadata
            # is unavailable.
            text = result.page_content.strip()

            if text:
                transcript_parts.append(
                    {
                        "timestamp_seconds": None,
                        "timestamp": None,
                        "text": text,
                    }
                )

    if not transcript_parts:
        raise ValueError(
            "The indexed video does not contain usable transcript text."
        )

    transcript_parts.sort(
        key=lambda item: (
            item["timestamp_seconds"]
            if item["timestamp_seconds"] is not None
            else float("inf")
        )
    )

    transcript = "\n".join(
        (
            f"[{item['timestamp']}] {item['text']}"
            if item["timestamp"]
            else item["text"]
        )
        for item in transcript_parts
    )

    # Keep the initial implementation within a practical
    # local-model context size.
    max_transcript_chars = 18000
    transcript = transcript[:max_transcript_chars]

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
    )

    # Generate study notes.
    notes_prompt = f"""
You are an expert educational tutor creating a high-quality
study guide from a YouTube video transcript.

Use ONLY the supplied transcript as the factual source.
Do not invent timestamps, examples, definitions, or claims.
If the transcript does not support a detail, omit it.

Create detailed, well-organized Markdown notes using these sections:

# Study Guide

## 1. Overview
Explain the central topic and learning objectives.

## 2. Key Concepts
Explain the important concepts in clear language.

## 3. Important Definitions
Define technical terms using the transcript.

## 4. Step-by-Step Explanation
Explain processes in the order presented in the transcript.

## 5. Examples
Include examples actually supported by the transcript.

## 6. Important Timestamps
Include only timestamps available in the transcript.
Explain briefly what each timestamp covers.

## 7. Key Takeaways
Summarize the most important learning points.

## 8. Revision Questions
Create five useful questions based on the transcript.
Do not invent facts to answer them.

Requirements:
- Use clear Markdown headings and bullet points.
- Explain technical concepts rather than merely listing them.
- Avoid repeating the same point across sections.
- Do not claim that you watched or understood content absent
  from the supplied transcript.

TRANSCRIPT:
{transcript}
"""

    notes_response = llm.invoke(notes_prompt)
    notes = str(notes_response.content).strip()

    if not notes:
        raise RuntimeError("The model returned empty study notes.")

    # Generate a separate flowchart.
    flowchart_prompt = f"""
You create Mermaid flowcharts for educational study guides.

Using ONLY the supplied transcript, create one useful flowchart
that explains a process or the relationship between concepts.

Return ONLY Mermaid code.
Do not use Markdown code fences.
Do not include explanations outside the diagram.

Rules:
- Begin with flowchart TD.
- Use simple node IDs such as A, B, C.
- Use quoted labels when they contain punctuation.
- Keep labels short and readable.
- Use only relationships supported by the transcript.
- Do not invent steps or connections.
- If the transcript does not describe a clear process,
  create a concept relationship diagram instead.
- Keep the diagram to approximately 5–12 nodes.

Example format:

flowchart TD
    A["Input"] --> B["Processing"]
    B --> C["Output"]

TRANSCRIPT:
{transcript}
"""

    flowchart_response = llm.invoke(flowchart_prompt)
    flowchart = str(flowchart_response.content).strip()

    # Remove code fences if the model adds them.
    if flowchart.startswith("```"):
        lines = flowchart.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        flowchart = "\n".join(lines).strip()

    # Basic validation before sending the diagram to the frontend.
    if not flowchart.startswith(
        ("flowchart ", "graph ")
    ):
        flowchart = (
            "flowchart TD\n"
            '    A["Study the main concepts"] --> '
            'B["Review the detailed notes"]'
        )

    return {
        "video_id": video_id,
        "notes": notes,
        "flowchart": flowchart,
        "transcript_segments_used": len(transcript_parts),
        "message": "Study guide generated successfully.",
    }