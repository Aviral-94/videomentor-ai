from langchain_core.documents import Document


CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def create_chunks(entries):
    chunks = []

    current_entries = []
    current_length = 0

    for entry in entries:

        current_entries.append(entry)
        current_length += len(entry["text"])

        if current_length >= CHUNK_SIZE:

            chunks.append(_create_document(current_entries))

            # Keep overlap
            overlap_entries = []
            overlap_length = 0

            for item in reversed(current_entries):

                overlap_entries.insert(0, item)
                overlap_length += len(item["text"])

                if overlap_length >= CHUNK_OVERLAP:
                    break

            current_entries = overlap_entries
            current_length = overlap_length

    if current_entries:
        chunks.append(_create_document(current_entries))

    return chunks


def _create_document(entries):

    text = " ".join(
        entry["text"]
        for entry in entries
    )

    # Preserve individual transcript timestamps
    segments = []

    for entry in entries:
        segments.append({
            "text": entry["text"],
            "start": entry["start"],
            "end": entry["end"]
        })

    return Document(
        page_content=text,
        metadata={
            "video_id": entries[0]["video_id"],
            "start": entries[0]["start"],
            "end": entries[-1]["end"],
            "segments": segments
        }
    )