import re
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import IpBlocked


def extract_video_id(url: str) -> str:
    """Extract a YouTube video ID from supported URL formats."""

    parsed = urlparse(url.strip())
    hostname = parsed.hostname or ""

    if hostname == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]

    elif hostname in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com",
    }:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]

        else:
            parts = parsed.path.strip("/").split("/")

            if len(parts) >= 2 and parts[0] in {
                "shorts",
                "embed",
                "live",
            }:
                video_id = parts[1]
            else:
                raise ValueError("Unsupported YouTube URL format.")

    else:
        raise ValueError("Please provide a valid YouTube URL.")

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Invalid YouTube video ID.")

    return video_id


def fetch_transcript(video_id: str):
    """
    Fetch a transcript using YouTubeTranscriptApi.

    Prefer English when available, then try other available languages.
    The API cannot retrieve a transcript if the video has none accessible.
    """

    api = YouTubeTranscriptApi()

    try:
        # Fetch the English transcript when possible.
        try:
            transcript = api.fetch(video_id, languages=["en"])
        except Exception as english_error:
            # If English is unavailable, inspect available transcripts.
            transcript_list = api.list(video_id)

            try:
                transcript = transcript_list.find_transcript(
                    ["en"]
                ).fetch()
            except Exception:
                try:
                    transcript = transcript_list.find_generated_transcript(
                        ["en"]
                    ).fetch()
                except Exception:
                    # Try any available transcript language.
                    available = list(transcript_list)

                    if not available:
                        raise RuntimeError(
                            "This video has no accessible transcript."
                        ) from english_error

                    transcript = available[0].fetch()

    except IpBlocked as exc:
        raise RuntimeError(
            "YouTube is blocking transcript requests from this IP. "
            "Please try again later."
        ) from exc

    except Exception as exc:
        raise RuntimeError(
            "Could not retrieve this video's transcript. "
            "The video may have captions disabled, no accessible "
            "transcript, or temporary YouTube restrictions. "
            f"Details: {exc}"
        ) from exc

    entries = []

    for item in transcript:
        text = item.text.replace("\n", " ").strip()

        if not text:
            continue

        entries.append({
            "text": text,
            "start": item.start,
            "end": item.start + item.duration,
            "video_id": video_id,
        })

    if not entries:
        raise RuntimeError(
            "The transcript was retrieved, but it contained no usable text."
        )

    return entries