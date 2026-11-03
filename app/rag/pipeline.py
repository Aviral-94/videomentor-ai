import re
from functools import lru_cache
from app.llm import get_llm
from langchain_core.documents import Document
from qdrant_client.http import models as qdrant_models

from app.vectorstore.qdrant import get_vector_store
from app.vectorstore.embeddings import get_embeddings
from app.rag.memory import get_history, add_message


TUTOR_MODES = {
    "explain": """
You are in EXPLAIN mode.
- Explain concepts in beginner-friendly language.
- Break complicated ideas into smaller steps.
- Use examples from the transcript when helpful.
- Explain technical terms simply.
- Focus on understanding rather than memorization.
""",
    "summarize": """
You are in SUMMARIZE mode.
- Identify the main ideas and present important points as bullets.
- Preserve important technical details and avoid repetition.
- Include timestamps when useful.
- Do not add information absent from the transcript.
""",
    "quiz": """
You are in QUIZ mode.
- Generate questions based on the video transcript.
- Prefer one question at a time unless multiple are requested.
- Ask a clear question without immediately revealing the answer.
- Evaluate the user's answer and explain what was right or missed.
""",
    "interview": """
You are in INTERVIEW mode.
- Ask interview-style questions based on the video.
- Prefer one question at a time and wait for the user's response.
- Evaluate correctness, clarity, and completeness.
- Give constructive feedback and explain an ideal answer when appropriate.
- Increase difficulty gradually when possible.
""",
}


def format_timestamp(seconds):
    """Convert seconds into M:SS or H:MM:SS format."""
    seconds = max(0, int(seconds or 0))
    if seconds >= 3600:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{seconds // 60}:{seconds % 60:02d}"


def _video_filter(video_id):
    """Build a Qdrant filter for LangChain's metadata.video_id field."""
    return qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="metadata.video_id",
                match=qdrant_models.MatchValue(value=video_id),
            )
        ]
    )


def _metadata(result):
    meta = getattr(result, "metadata", None)
    return meta if isinstance(meta, dict) else {}


def _fallback_segment(result):
    meta = _metadata(result)
    text = (getattr(result, "page_content", "") or "").strip()
    if not text:
        return None
    start = meta.get("start", 0)
    end = meta.get("end", start)
    try:
        start = float(start or 0)
        end = float(end or start)
    except (TypeError, ValueError):
        start, end = 0.0, 0.0
    return {"start": start, "end": end, "text": text}


def get_nearby_segments(results, current_time, window=30):
    """Find stored transcript segments near the given timestamp."""
    nearby = []
    current_time = float(current_time or 0)
    for result in results:
        meta = _metadata(result)
        segments = meta.get("segments") or []
        if segments:
            for seg in segments:
                try:
                    st = float(seg.get("start", 0))
                    en = float(seg.get("end", st))
                except (TypeError, ValueError):
                    continue
                if st <= current_time + window and en >= current_time - window:
                    nearby.append(seg)
        else:
            fb = _fallback_segment(result)
            if fb and fb["start"] <= current_time + window and fb["end"] >= current_time - window:
                nearby.append(fb)
    return nearby


def _cosine_similarity(a, b):
    mag_a = sum(v * v for v in a) ** 0.5
    mag_b = sum(v * v for v in b) ** 0.5
    if not mag_a or not mag_b:
        return -1.0
    return sum(x * y for x, y in zip(a, b)) / (mag_a * mag_b)


def _tokens(text):
    return {token.lower() for token in re.findall(r"[\w]+", text or "", flags=re.UNICODE) if len(token) > 2}


def _extract_requested_timestamp(question):
    """Return an explicitly requested video timestamp in seconds, if present."""
    text = (question or "").strip()
    # Match H:MM:SS
    match = re.search(r'(?<!\d)(\d{1,2}):([0-5]\d):([0-5]\d)(?!\d)', text)
    if match:
        h, m, s = map(int, match.groups())
        return h * 3600 + m * 60 + s
    # Match M:SS or MM:SS
    match = re.search(r'(?<!\d)(\d{1,3}):([0-5]\d)(?!\d)', text)
    if match:
        m, s = map(int, match.groups())
        return m * 60 + s
    # Match textual descriptions e.g. "at 45 minutes" or "around 1 hour 15 minutes"
    h_m_s = re.search(r'\b(?:at|around|from|near)?\s*(?:(\d+)\s*(?:hours?|hrs?|h))?\s*(?:(\d+)\s*(?:minutes?|mins?|m))?\s*(?:(\d+)\s*(?:seconds?|secs?|s))?\b', text, re.I)
    if h_m_s and any(h_m_s.groups()):
        h = int(h_m_s.group(1) or 0)
        m = int(h_m_s.group(2) or 0)
        s = int(h_m_s.group(3) or 0)
        total = h * 3600 + m * 60 + s
        if total > 0:
            return total
    return None


def _is_overview_question(question):
    """Check if the user is asking a broad question about the video's main ideas or outline."""
    lower = (question or "").lower()
    triggers = [
        "three main ideas", "main ideas", "main topics", "key topics",
        "what is this video about", "what does this video cover",
        "overview of this video", "overview of the video",
        "summarize this video", "summarize the video", "summary of this video",
        "what are the topics", "what topics are discussed",
    ]
    return any(t in lower for t in triggers)


def _get_all_video_chunks(vector_store, video_id):
    """Scroll and collect all stored chunks for a video, ordered by start time."""
    points = []
    offset = None
    try:
        while True:
            batch, next_offset = vector_store.client.scroll(
                collection_name=vector_store.collection_name,
                scroll_filter=_video_filter(video_id),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset
    except Exception as exc:
        print(f"[VideoMentor] Error scrolling chunks for {video_id}: {exc}")
        return []

    docs = []
    for p in points:
        payload = p.payload or {}
        raw_meta = payload.get("metadata") or {}
        meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        meta.setdefault("video_id", payload.get("video_id", video_id))
        meta.setdefault("start", payload.get("start", 0))
        meta.setdefault("end", payload.get("end", 0))
        content = (
            payload.get("page_content") or payload.get("text")
            or meta.get("page_content") or meta.get("text") or ""
        )
        if content:
            docs.append(Document(page_content=str(content), metadata=meta))

    docs.sort(key=lambda d: float(_metadata(d).get("start", 0) or 0))
    return docs


def _expand_adjacent_chunks(seed_docs, all_video_chunks, max_adjacent=1):
    """Expand retrieved seed chunks with immediately adjacent chunks from the same video."""
    if not all_video_chunks or not seed_docs:
        return seed_docs

    index_map = {}
    for i, d in enumerate(all_video_chunks):
        st = float(_metadata(d).get("start", 0) or 0)
        index_map[round(st, 2)] = i

    expanded_indices = set()
    for d in seed_docs:
        st = float(_metadata(d).get("start", 0) or 0)
        idx = index_map.get(round(st, 2))
        if idx is not None:
            for offset in range(-max_adjacent, max_adjacent + 1):
                target_idx = idx + offset
                if 0 <= target_idx < len(all_video_chunks):
                    expanded_indices.add(target_idx)

    return [all_video_chunks[i] for i in sorted(expanded_indices)]


def _retrieve_for_video(vector_store, query, video_id, k=5):
    """Retrieve top-k chunks for a single video using Qdrant filter with Python fallback."""
    try:
        results = vector_store.similarity_search(
            query,
            k=k,
            filter=_video_filter(video_id),
        )
    except Exception as exc:
        print(f"[VideoMentor] Filtered search error: {exc}")
        results = []

    filtered = [r for r in results if _metadata(r).get("video_id") == video_id]
    if filtered:
        return filtered

    # Fallback search if filter failed
    try:
        broad = vector_store.similarity_search(query, k=max(20, k * 5))
        return [r for r in broad if _metadata(r).get("video_id") == video_id][:k]
    except Exception as exc:
        print(f"[VideoMentor] Fallback search error: {exc}")
        return []


def resolve_followup_question(question, history, llm):
    """
    Rewrite conversational follow-ups into standalone, topic-specific retrieval queries.
    Preserves referents like 'the second one', 'it', 'that example', etc.
    """
    if not history:
        return question

    lower = question.lower().strip()
    is_followup = (
        any(w in lower for w in [
            "it", "that", "this", "them", "second", "2nd", "first", "1st", "third", "3rd",
            "last", "the example", "what example", "which example", "explain more",
            "why", "how", "where", "can you explain", "immediately after", "tell me more"
        ])
        or len(question.split()) <= 6
    )

    if not is_followup:
        return question

    recent_history_text = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}"
        for m in history[-6:]
    )

    prompt = f"""Given the recent conversation history, rewrite the user's latest follow-up question into a concise standalone search query for video transcript retrieval.
Rules:
- Resolve references like "it", "that", "this", "the second one", "what example", "that example" to the specific concept or topic being discussed in the conversation.
- If referencing a numbered item from the previous answer (e.g. "the second one"), identify that specific concept name.
- Do not answer the question.
- Do not add conversational words.
- Output ONLY the standalone search query on a single line.

Recent Conversation:
{recent_history_text}

User Follow-Up:
{question}

Standalone Query:"""

    try:
        resp = llm.invoke(prompt)
        clean = str(resp.content).strip().strip('"').strip("'")
        first_line = clean.split("\n")[0].strip()
        if first_line and len(first_line) > 3:
            return first_line
    except Exception as exc:
        print(f"[VideoMentor] LLM followup resolution failed: {exc}")

    # Fallback heuristic: check for numbered reference in last assistant answer
    last_assistant = next(
        (m.get("content", "") for m in reversed(history) if m.get("role") == "assistant"),
        ""
    )
    if "second" in lower or "2nd" in lower:
        items = re.findall(r"(?m)^\s*2[.)]\s*(?:\*\*)?([^\n*]+?)(?:\*\*)?\s*(?:$|:)", last_assistant)
        if items:
            return f"{items[0].strip()} in video"
    elif "first" in lower or "1st" in lower:
        items = re.findall(r"(?m)^\s*1[.)]\s*(?:\*\*)?([^\n*]+?)(?:\*\*)?\s*(?:$|:)", last_assistant)
        if items:
            return f"{items[0].strip()} in video"

    return question


def find_best_source(question, results):
    """Backwards-compatible helper to find the most relevant source from a list of results."""
    if not results:
        return None
    embeddings = get_embeddings()
    q_emb = embeddings.embed_query(question)
    best_doc = None
    best_score = -1.0
    for doc in results:
        text = doc.page_content.strip()
        if not text:
            continue
        try:
            score = _cosine_similarity(q_emb, embeddings.embed_query(text))
            if score > best_score:
                best_score = score
                best_doc = doc
        except Exception:
            continue
    if best_doc:
        meta = _metadata(best_doc)
        return {
            "video_id": meta.get("video_id"),
            "start": meta.get("start", 0),
            "text": best_doc.page_content[:240],
        }
    return None


def find_best_segment(question, results):
    """Backwards-compatible helper to find the best segment."""
    src = find_best_source(question, results)
    if src:
        return {"start": src["start"], "text": src["text"]}
    return None


def ask_video(
    question: str,
    video_id: str,
    current_time: float,
    conversation_id: str,
    mode: str = "explain",
    video_ids: list[str] | None = None,
):
    """
    Answer questions about YouTube video transcript(s) using strictly grounded evidence.
    Supports history-aware follow-up resolution, adjacent-chunk expansion,
    accurate timestamp retrieval, and evidence-based source selection.
    """
    question = (question or "").strip()
    mode = (mode or "explain").lower().strip()
    if mode not in TUTOR_MODES:
        mode = "explain"

    selected_video_ids = []
    for cid in (video_ids or [video_id]):
        if isinstance(cid, str):
            cid = cid.strip()
            if cid and cid not in selected_video_ids:
                selected_video_ids.append(cid)
    if not selected_video_ids and video_id:
        selected_video_ids = [video_id]

    if not selected_video_ids:
        return {
            "answer": "Please load a YouTube video before asking a question.",
            "timestamp": "0:00",
            "timestamp_seconds": 0,
            "youtube_url": "",
            "mode": mode,
            "sources": [],
        }

    primary_video_id = video_id if video_id in selected_video_ids else selected_video_ids[0]
    is_multi_video = len(selected_video_ids) > 1

    history = get_history(conversation_id)
    llm = get_llm(temperature=0)
    embeddings = get_embeddings()
    vector_store = get_vector_store()

    # Step 1: Follow-up resolution
    retrieval_query = resolve_followup_question(question, history, llm)
    print(f"[VideoMentor][followup-resolution] original='{question}' -> resolved='{retrieval_query}'")

    requested_timestamp = _extract_requested_timestamp(question)
    is_overview = _is_overview_question(question)

    # Step 2: Retrieve chunks per video with adjacent-chunk expansion
    evidence_docs_by_video = {}
    all_evidence_docs = []

    for vid in selected_video_ids:
        all_chunks = _get_all_video_chunks(vector_store, vid)
        if not all_chunks:
            # Fallback to direct similarity search
            all_chunks = _retrieve_for_video(vector_store, "video", vid, k=20)

        if requested_timestamp is not None:
            # Strict timestamp-specific retrieval
            target = requested_timestamp
            near_chunks = [
                d for d in all_chunks
                if abs(float(_metadata(d).get("start", 0)) - target) <= 60
            ]
            evidence_docs = near_chunks
            print(f"[VideoMentor][timestamp-retrieval] video={vid} target={target}s matched={len(near_chunks)}")

        elif is_overview:
            # Broad overview question: include intro/outline chunks (first 180s) + top semantic matches
            intro_chunks = [d for d in all_chunks if float(_metadata(d).get("start", 0)) <= 150]
            semantic_chunks = _retrieve_for_video(vector_store, retrieval_query, vid, k=6)
            merged = {round(float(_metadata(d).get("start", 0)), 2): d for d in intro_chunks}
            for d in semantic_chunks:
                merged[round(float(_metadata(d).get("start", 0)), 2)] = d
            evidence_docs = sorted(merged.values(), key=lambda d: float(_metadata(d).get("start", 0)))
            print(f"[VideoMentor][overview-retrieval] video={vid} total_evidence={len(evidence_docs)}")

        else:
            # Semantic search with query variants and adjacent-chunk expansion
            seed_docs = _retrieve_for_video(vector_store, retrieval_query, vid, k=4)
            if retrieval_query != question:
                seed_docs.extend(_retrieve_for_video(vector_store, question, vid, k=2))

            # Deduplicate seeds
            seen_seeds = set()
            unique_seeds = []
            for d in seed_docs:
                st = round(float(_metadata(d).get("start", 0)), 2)
                if st not in seen_seeds:
                    seen_seeds.add(st)
                    unique_seeds.append(d)

            # Expand with immediate adjacent chunks from the same video
            evidence_docs = _expand_adjacent_chunks(unique_seeds, all_chunks, max_adjacent=1)
            print(f"[VideoMentor][semantic-retrieval] video={vid} seeds={len(unique_seeds)} expanded={len(evidence_docs)}")

        evidence_docs_by_video[vid] = evidence_docs
        all_evidence_docs.extend(evidence_docs)

    # Step 3: Check if evidence is available
    if not all_evidence_docs:
        if requested_timestamp is not None:
            answer = (
                f"The video transcript does not contain any indexed speech or captions around "
                f"{format_timestamp(requested_timestamp)}. No transcript evidence is available for that time window."
            )
        else:
            answer = (
                "The retrieved transcript does not contain enough information to answer this question. "
                "Please verify that the video transcript has been indexed."
            )
        add_message(conversation_id, "user", question, video_id=primary_video_id)
        add_message(conversation_id, "assistant", answer, video_id=primary_video_id)
        sec = int(requested_timestamp if requested_timestamp is not None else current_time)
        return {
            "answer": answer,
            "timestamp": format_timestamp(sec),
            "timestamp_seconds": sec,
            "youtube_url": f"https://www.youtube.com/watch?v={primary_video_id}&t={sec}s",
            "mode": mode,
            "sources": [],
        }

    # Step 4: Build evidence context
    context_parts = []
    for i, doc in enumerate(all_evidence_docs, 1):
        meta = _metadata(doc)
        source_vid = meta.get("video_id", primary_video_id)
        st = float(meta.get("start", 0))
        end = float(meta.get("end", st))
        text = doc.page_content.strip()
        context_parts.append(
            f"[Evidence {i} | video_id={source_vid} | start={format_timestamp(st)} ({int(st)}s) | end={format_timestamp(end)} ({int(end)}s)]\n"
            f"{text}"
        )
    context = "\n\n".join(context_parts)

    history_text = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}"
        for m in history[-6:]
    )

    source_instructions = (
        "Identify video sources only by their supplied IDs and keep claims from different videos distinct."
        if is_multi_video else
        "Use only the supplied transcript evidence for claims about this video."
    )

    prompt = f"""You are VideoMentor, an AI tutor that answers questions about YouTube videos.

TUTOR MODE:
{TUTOR_MODES[mode]}

SELECTED VIDEO IDS:
{', '.join(selected_video_ids)}

RELEVANT TRANSCRIPT EVIDENCE:
{context}

CONVERSATION HISTORY:
{history_text or '(No earlier conversation)'}

CURRENT USER QUESTION:
{question}

STANDALONE QUERY CONTEXT:
{retrieval_query}

STRICT ANSWER RULES:
- Use ONLY the supplied transcript evidence as the source of truth.
- Answer the user's question directly, clearly, and factually based on the evidence.
- If asked what example the speaker gives, identify the specific example, scenario, or model explicitly described by the speaker in the evidence.
- If the evidence does not contain sufficient information or a specific example, clearly state: "The retrieved transcript does not contain enough information to identify a specific example given by the speaker."
- Do not fabricate quotations, examples, or speaker statements.
- Do not invent facts or draw external assumptions beyond what is in the evidence.
- If the user asks about an explicit timestamp, answer only from the transcript passages around that timestamp.
- For follow-up questions referencing "the second one", "it", or "that", maintain topic continuity and explain how the speaker discusses that topic.
- {source_instructions}
- Do not mention internal processes, embeddings, RAG, or model names.

ANSWER:"""

    resp = llm.invoke(prompt)
    answer = str(resp.content).strip()
    print(f"[VideoMentor][answer] {answer[:400]}...")

    add_message(conversation_id, "user", question, video_id=primary_video_id)
    add_message(conversation_id, "assistant", answer, video_id=primary_video_id)

    # Step 5: Evidence-based source selection & validation
    answer_tokens = _tokens(answer)
    stop_words = {
        "the", "and", "that", "this", "with", "from", "what", "when", "where",
        "which", "does", "how", "about", "according", "transcript", "video",
        "explain", "explained", "please", "answer", "speaker", "mention", "mentions"
    }
    answer_tokens -= stop_words

    answer_emb = []
    try:
        answer_emb = embeddings.embed_query(answer[:600]) if answer else []
    except Exception:
        pass

    # Batch embed candidate texts in a single forward pass
    candidate_texts = [doc.page_content.strip()[:400] for doc in all_evidence_docs]
    doc_embs = []
    if answer_emb and candidate_texts:
        try:
            doc_embs = embeddings.embed_documents(candidate_texts)
        except Exception:
            doc_embs = [[] for _ in candidate_texts]

    scored_candidates = []
    for i, doc in enumerate(all_evidence_docs):
        meta = _metadata(doc)
        source_vid = meta.get("video_id", primary_video_id)
        st = float(meta.get("start", 0))
        text = doc.page_content.strip()
        doc_toks = _tokens(text)

        overlap = len(answer_tokens & doc_toks) / max(1, len(answer_tokens))
        sem_score = 0.0
        if answer_emb and i < len(doc_embs) and doc_embs[i]:
            try:
                sem_score = _cosine_similarity(answer_emb, doc_embs[i])
            except Exception:
                pass

        total_score = 0.5 * sem_score + 0.5 * overlap
        scored_candidates.append({
            "video_id": source_vid,
            "start": st,
            "text": text,
            "score": total_score,
            "overlap": overlap,
        })

    # Sort candidates by score
    scored_candidates.sort(key=lambda s: s["score"], reverse=True)

    sources = []
    used_windows = []  # tuple of (vid, start)

    # Only include sources if there is meaningful relevance
    is_insufficient = "not contain enough information" in answer.lower()

    if not is_insufficient:
        for c in scored_candidates:
            st = c["start"]
            c_vid = c["video_id"]
            # Avoid duplicate or near-identical timestamps from same video (within 30s)
            if any(c_vid == u_vid and abs(st - u_st) < 30 for u_vid, u_st in used_windows):
                continue
            used_windows.append((c_vid, st))
            seconds = int(st)
            sources.append({
                "video_id": c_vid,
                "timestamp": format_timestamp(seconds),
                "timestamp_seconds": seconds,
                "youtube_url": f"https://www.youtube.com/watch?v={c_vid}&t={seconds}s",
                "excerpt": c["text"][:240],
            })
            if len(sources) >= 3:
                break

    primary_sec = sources[0]["timestamp_seconds"] if sources else (
        int(requested_timestamp) if requested_timestamp is not None else int(current_time or 0)
    )
    primary_ts = format_timestamp(primary_sec)

    return {
        "answer": answer,
        "timestamp": primary_ts,
        "timestamp_seconds": primary_sec,
        "youtube_url": f"https://www.youtube.com/watch?v={primary_video_id}&t={primary_sec}s",
        "mode": mode,
        "sources": sources,
    }
