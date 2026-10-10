from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import IpBlocked


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from a URL."""

    if "v=" in url:
        return url.split("v=")[1].split("&")[0]

    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]

    raise ValueError("Invalid YouTube URL")


def fetch_transcript(video_id: str):
    """Fetch and normalize a YouTube transcript."""

    api = YouTubeTranscriptApi()

    try:
        transcript = api.fetch(video_id)

    except IpBlocked:
        raise RuntimeError(
            "YouTube is currently blocking transcript requests "
            "from this IP. Please try again later."
        )

    entries = []

    for item in transcript:

        text = item.text.replace("\n", " ").strip()

        if not text:
            continue

        entries.append({
            "text": text,
            "start": item.start,
            "end": item.start + item.duration,
            "video_id": video_id
        })

    return entries