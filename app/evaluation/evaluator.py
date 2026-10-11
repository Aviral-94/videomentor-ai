from app.vectorstore.qdrant import get_vector_store
from app.evaluation.dataset import EVALUATION_DATASET


def evaluate_retrieval(test_case):
    """
    Evaluate retrieval using keyword coverage and
    a simple question-to-context relevance proxy.
    """

    video_id = test_case["video_id"]
    question = test_case["question"]
    expected_keywords = test_case["expected_keywords"]

    vector_store = get_vector_store()

    results = vector_store.similarity_search(
        question,
        k=3,
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

    retrieved_text = " ".join(
        result.page_content for result in results
    ).lower()

    # 1. Keyword coverage
    matched_keywords = [
        keyword
        for keyword in expected_keywords
        if keyword.lower() in retrieved_text
    ]

    keyword_coverage = (
        len(matched_keywords) / len(expected_keywords) * 100
        if expected_keywords
        else 0
    )

    # 2. Simple relevance proxy:
    # Count how many meaningful question words occur
    # in the retrieved text.
    stop_words = {
        "what", "is", "are", "was", "were",
        "the", "a", "an", "in", "on", "of",
        "to", "for", "does", "do", "how",
        "and", "or", "their", "its", "it",
    }

    question_words = {
        word.strip(".,?!:;()[]{}\"'").lower()
        for word in question.split()
        if word.lower() not in stop_words
        and len(word.strip(".,?!:;()[]{}\"'")) > 2
    }

    matched_question_words = [
        word
        for word in question_words
        if word in retrieved_text
    ]

    relevance_proxy = (
        len(matched_question_words) / len(question_words) * 100
        if question_words
        else 0
    )

    return {
        "test_id": test_case["id"],
        "question": question,
        "video_id": video_id,
        "retrieved_chunks": len(results),
        "matched_keywords": matched_keywords,
        "keyword_coverage": round(keyword_coverage, 2),
        "question_words_matched": matched_question_words,
        "relevance_proxy": round(relevance_proxy, 2),
        "passed": keyword_coverage >= 50,
    }


def run_evaluation():
    """Evaluate every test case and summarize the results."""

    results = []

    for test_case in EVALUATION_DATASET:
        try:
            result = evaluate_retrieval(test_case)
        except Exception as error:
            result = {
                "test_id": test_case["id"],
                "question": test_case["question"],
                "keyword_coverage": 0,
                "relevance_proxy": 0,
                "passed": False,
                "error": str(error),
            }

        results.append(result)

    total_tests = len(results)

    passed_tests = sum(
        1 for result in results if result["passed"]
    )

    average_keyword_coverage = (
        sum(
            result["keyword_coverage"]
            for result in results
        ) / total_tests
        if total_tests
        else 0
    )

    average_relevance_proxy = (
        sum(
            result["relevance_proxy"]
            for result in results
        ) / total_tests
        if total_tests
        else 0
    )

    return {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "average_keyword_coverage": round(
            average_keyword_coverage, 2
        ),
        "average_relevance_proxy": round(
            average_relevance_proxy, 2
        ),
        "results": results,
    }


if __name__ == "__main__":
    import json

    print(
        json.dumps(
            run_evaluation(),
            indent=2,
            ensure_ascii=False,
        )
    )