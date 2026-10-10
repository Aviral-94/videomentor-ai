from app.rag.pipeline import ask_video


def main():

    print("================================")
    print("        VideoMentor Chat")
    print("================================")

    video_id = input("\nEnter Video ID: ").strip()

    current_time = float(
        input("Enter current video time in seconds: ").strip()
    )

    print("\nVideo loaded from Qdrant.")
    print("You can now ask questions.")
    print("Type 'exit' to quit.")

    while True:

        question = input("\nQuestion: ")

        if question.lower() == "exit":
            print("\nGoodbye! 👋")
            break

        if not question.strip():
            continue

        try:

            result = ask_video(
                question,
                video_id,
                current_time
            )

            print("\n===== ANSWER =====")
            print(result["answer"])

            print(
                f"\n📍 Relevant moment: "
                f"{result['timestamp']}"
            )

            print(
                f"▶ Watch from "
                f"{result['timestamp']}:"
            )

            print(result["youtube_url"])

        except Exception as error:

            print("\n❌ Error while answering:")
            print(error)


if __name__ == "__main__":
    main()