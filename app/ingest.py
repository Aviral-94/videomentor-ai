from app.ingestion.youtube import extract_video_id, fetch_transcript
from app.ingestion.chunker import create_chunks
from app.vectorstore.qdrant import store_documents


def ingest_video(url):

    print("\n[1/4] Extracting video ID...")

    video_id = extract_video_id(url)

    print("Video ID:", video_id)

    print("\n[2/4] Fetching transcript...")

    entries = fetch_transcript(video_id)

    print("Transcript entries:", len(entries))

    print("\n[3/4] Creating chunks...")

    chunks = create_chunks(entries)

    print("Chunks:", len(chunks))

    print("\n[4/4] Storing in Qdrant...")

    store_documents(chunks)

    print("\n✅ Video successfully ingested!")

    return video_id


def main():

    print("================================")
    print("      VideoMentor Ingestion")
    print("================================")

    url = input("\nYouTube URL: ")

    try:

        ingest_video(url)

    except Exception as error:

        print("\n❌ Ingestion Error:")
        print(error)


if __name__ == "__main__":
    main()