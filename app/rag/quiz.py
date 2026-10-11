import json
import re

from langchain_ollama import ChatOllama
from app.vectorstore.qdrant import get_vector_store


QUIZ_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 4,
                        "maxItems": 4
                    },
                    "correct_answer": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 3
                    },
                    "explanation": {"type": "string"},
                    "timestamp_seconds": {
                        "type": "integer",
                        "minimum": 0
                    }
                },
                "required": [
                    "question",
                    "options",
                    "correct_answer",
                    "explanation",
                    "timestamp_seconds"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": ["questions"],
    "additionalProperties": False
}


def parse_json_response(response):
    """Parse a JSON object returned by Ollama."""

    output = str(response.content).strip()

    # Remove Markdown fences if present.
    output = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        output,
        flags=re.IGNORECASE
    )

    output = re.sub(r"\s*```\s*$", "", output)

    start = output.find("{")

    if start == -1:
        raise ValueError("The AI did not return a JSON object.")

    depth = 0
    in_string = False
    escaped = False
    end = -1

    for index in range(start, len(output)):
        char = output[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False

            continue

        if char == '"':
            in_string = True

        elif char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:
                end = index + 1
                break

    if end == -1:
        raise ValueError("The AI returned incomplete JSON.")

    return json.loads(output[start:end])


def validate_quiz(data, num_questions):
    """Validate quiz structure and reject duplicate options."""

    if not isinstance(data, dict):
        raise ValueError("Invalid quiz response.")

    questions = data.get("questions")

    if not isinstance(questions, list):
        raise ValueError("The AI returned malformed quiz data.")

    if len(questions) != num_questions:
        raise ValueError(
            f"Expected {num_questions} questions, "
            f"but received {len(questions)}."
        )

    validated = []

    for index, question in enumerate(questions):
        number = index + 1

        if not isinstance(question, dict):
            raise ValueError(f"Question {number} has an invalid format.")

        question_text = question.get("question")
        options = question.get("options")
        answer = question.get("correct_answer")
        explanation = question.get("explanation")
        timestamp = question.get("timestamp_seconds")

        if not isinstance(question_text, str) or not question_text.strip():
            raise ValueError(f"Question {number} has no question text.")

        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError(f"Question {number} has no explanation.")

        if not isinstance(options, list) or len(options) != 4:
            raise ValueError(
                f"Question {number} must have exactly four options."
            )

        if not all(
            isinstance(option, str) and option.strip()
            for option in options
        ):
            raise ValueError(
                f"Question {number} contains an empty option."
            )

        normalized_options = [
            re.sub(r"\s+", " ", option).strip().casefold()
            for option in options
        ]

        if len(set(normalized_options)) != 4:
            raise ValueError(
                f"Question {number} contains duplicate options."
            )

        if (
            isinstance(answer, bool)
            or not isinstance(answer, int)
            or answer not in range(4)
        ):
            raise ValueError(
                f"Question {number} has an invalid answer index."
            )

        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or timestamp < 0
        ):
            raise ValueError(
                f"Question {number} has an invalid timestamp."
            )

        validated.append({
            "question": question_text.strip(),
            "options": [option.strip() for option in options],
            "correct_answer": answer,
            "explanation": explanation.strip(),
            "timestamp_seconds": timestamp
        })

    return validated


def generate_quiz(
    video_id: str,
    num_questions: int = 5,
    difficulty: str = "medium"
):
    """
    Generate and review a multiple-choice quiz using
    transcript chunks stored in Qdrant.
    """

    # --------------------------------------------------
    # 1. Validate input
    # --------------------------------------------------

    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("A valid video ID is required.")

    if num_questions not in (5, 10, 15):
        raise ValueError(
            "Number of questions must be 5, 10, or 15."
        )

    if not isinstance(difficulty, str):
        raise ValueError(
            "Difficulty must be easy, medium, or hard."
        )

    difficulty = difficulty.strip().lower()

    if difficulty not in ("easy", "medium", "hard"):
        raise ValueError(
            "Difficulty must be easy, medium, or hard."
        )

    # --------------------------------------------------
    # 2. Retrieve transcript chunks
    # --------------------------------------------------

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        "important concepts, definitions, explanations, "
        "examples, and key information from the video",
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

    if not documents:
        raise ValueError(
            "No transcript was found for this video. "
            "Please process or index the video first."
        )

    # --------------------------------------------------
    # 3. Build timestamp-aware transcript context
    # --------------------------------------------------

    transcript_parts = []

    for document in documents:
        metadata = document.metadata or {}
        text = document.page_content.strip()

        if not text:
            continue

        timestamp = metadata.get("start", 0)

        try:
            timestamp = max(0, int(float(timestamp)))
        except (TypeError, ValueError):
            timestamp = 0

        transcript_parts.append(
            f"[Timestamp: {timestamp} seconds]\n{text}"
        )

    transcript = "\n\n".join(transcript_parts)
    transcript = transcript[:6000]

    if not transcript.strip():
        raise ValueError(
            "The retrieved transcript contains no usable text."
        )

    # --------------------------------------------------
    # 4. Initialize local Ollama model
    # --------------------------------------------------

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
        format=QUIZ_SCHEMA
    )

    # --------------------------------------------------
    # 5. Generate initial quiz
    # --------------------------------------------------

    generation_prompt = f"""
You are an expert educational assessment creator.

Create exactly {num_questions} multiple-choice questions
using ONLY the transcript provided below.

DIFFICULTY: {difficulty}

TRANSCRIPT:
{transcript}

RULES:

1. Generate exactly {num_questions} questions.
2. Test important concepts actually explained in the transcript.
3. Make every question clear, specific, and unambiguous.
4. Do not repeat questions or test the same concept repeatedly.
5. Every question must have exactly four distinct options.
6. Exactly one option must be correct.
7. Avoid options that mean essentially the same thing.
8. Incorrect options must be plausible but clearly incorrect.
9. Do not make an option correct merely because it sounds
   technically sophisticated.
10. The correct_answer field is a zero-based index:
    0 = first option
    1 = second option
    2 = third option
    3 = fourth option
11. The explanation must directly justify the selected answer.
12. Check that the answer index points to the option described
    by the explanation.
13. Use timestamps from the transcript context.
14. Do not invent facts or claim the transcript says something
    that it does not say.

Difficulty:
- easy: definitions and basic understanding
- medium: conceptual understanding and applications
- hard: deeper reasoning about concepts explained in the video

Return only a JSON object matching the required schema.
"""

    try:
        response = llm.invoke(generation_prompt)
        initial_data = parse_json_response(response)

        initial_questions = validate_quiz(
            initial_data,
            num_questions
        )

    except ValueError as error:
        print(f"Initial quiz generation failed: {error}")

        raise ValueError(
            "The AI generated an invalid quiz. Please try again."
        ) from error

    except Exception as error:
        print(f"Ollama generation error: {error}")

        raise RuntimeError(
            "Failed to generate the quiz. Check that Ollama "
            "is running and llama3.2:3b is available."
        ) from error

    # --------------------------------------------------
    # 6. Second-pass answer verification
    # --------------------------------------------------

    quiz_json = json.dumps(
        {"questions": initial_questions},
        ensure_ascii=False
    )

    verification_prompt = f"""
You are an independent educational quiz reviewer.

Review every question against the supplied transcript.

TRANSCRIPT:
{transcript}

QUIZ TO REVIEW:
{quiz_json}

Your job is to return a corrected version of the quiz.

For EACH question:

1. Check whether the question is supported by the transcript.
2. Check every option against the transcript.
3. Confirm that exactly one option is correct.
4. Check that correct_answer points to the genuinely correct option.
5. Check whether the explanation proves that answer.
6. Correct explanations that are vague, contradictory, or inaccurate.
7. Correct answer indices if they point to the wrong option.
8. Replace duplicate or nearly identical options.
9. Replace questions that cannot be answered from the transcript.
10. Keep the original question count: {num_questions}.
11. Preserve the requested difficulty: {difficulty}.
12. Keep timestamps non-negative and close to the relevant concept.
13. Do not introduce facts unsupported by the transcript.
14. If a question is flawed, fix it rather than simply repeating it.

IMPORTANT:
Do not trust the original answer indices.
Independently evaluate each question and its options.

Return ONLY a JSON object using the same schema.
Do not include commentary or Markdown fences.
"""

    try:
        review_response = llm.invoke(verification_prompt)

        reviewed_data = parse_json_response(review_response)

        final_questions = validate_quiz(
            reviewed_data,
            num_questions
        )

    except ValueError as error:
        print(f"Quiz verification failed: {error}")

        # Do not silently return an unverified quiz.
        raise ValueError(
            "The AI could not verify the generated quiz. "
            "Please try generating it again."
        ) from error

    except Exception as error:
        print(f"Ollama verification error: {error}")

        raise RuntimeError(
            "Quiz verification failed. Check the backend terminal."
        ) from error

    # --------------------------------------------------
    # 7. Return the reviewed quiz
    # --------------------------------------------------

    return {
        "video_id": video_id,
        "difficulty": difficulty,
        "num_questions": num_questions,
        "questions": final_questions
    }