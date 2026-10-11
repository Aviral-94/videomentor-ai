import { useEffect, useState } from "react";

const API_URL = "http://127.0.0.1:8000";
const HISTORY_KEY = "videomentor-quiz-history";

function loadQuizHistory() {
  try {
    const saved = localStorage.getItem(HISTORY_KEY);

    if (!saved) return [];

    const parsed = JSON.parse(saved);

    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.error("Could not load quiz history:", error);
    return [];
  }
}

function formatDate(date) {
  return new Date(date).toLocaleString([], {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export default function Quiz({ videoId, onJumpToTimestamp }) {
  const [difficulty, setDifficulty] = useState("medium");
  const [questionCount, setQuestionCount] = useState(5);

  const [questions, setQuestions] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedAnswers, setSelectedAnswers] = useState({});

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [finished, setFinished] = useState(false);

  const [history, setHistory] = useState(loadQuizHistory);
  const [showHistory, setShowHistory] = useState(false);

  // Keep the browser's saved history synchronized with React state.
  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch (error) {
      console.error("Could not save quiz history:", error);
      setError(
        "Quiz history could not be saved. Your browser storage may be full."
      );
    }
  }, [history]);

  async function generateQuiz() {
    if (!videoId || loading) return;

    setLoading(true);
    setError("");
    setQuestions([]);
    setCurrentIndex(0);
    setSelectedAnswers({});
    setFinished(false);
    setShowHistory(false);

    try {
      const response = await fetch(`${API_URL}/api/quiz`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          video_id: videoId,
          num_questions: questionCount,
          difficulty,
        }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        const detail =
          typeof data.detail === "string"
            ? data.detail
            : `Quiz request failed (${response.status}).`;

        throw new Error(detail);
      }

      if (
        !Array.isArray(data.questions) ||
        data.questions.length === 0
      ) {
        throw new Error("The backend returned no quiz questions.");
      }

      setQuestions(data.questions);
    } catch (err) {
      setError(
        `${err.message || "Could not generate the quiz."} ` +
          "Check that FastAPI, Ollama, and Qdrant are running."
      );
    } finally {
      setLoading(false);
    }
  }

  function selectAnswer(optionIndex) {
    if (
      selectedAnswers[currentIndex] !== undefined ||
      finished
    ) {
      return;
    }

    setSelectedAnswers((previous) => ({
      ...previous,
      [currentIndex]: optionIndex,
    }));
  }

  function nextQuestion() {
    if (selectedAnswers[currentIndex] === undefined) {
      return;
    }

    if (currentIndex < questions.length - 1) {
      setCurrentIndex((previous) => previous + 1);
      return;
    }

    // The final question has been answered.
    const finalScore = questions.reduce(
      (total, question, index) =>
        total +
        (
          selectedAnswers[index] === question.correct_answer
            ? 1
            : 0
        ),
      0
    );

    const attempt = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      videoId,
      difficulty,
      questionCount: questions.length,
      score: finalScore,
      percentage: Math.round(
        (finalScore / questions.length) * 100
      ),
      completedAt: new Date().toISOString(),

      // Store questions and answers for later review.
      questions: questions.map((question) => ({
        ...question,
        options: [...question.options],
      })),

      selectedAnswers: { ...selectedAnswers },
    };

    setHistory((previous) => [attempt, ...previous]);
    setFinished(true);
  }

  function restartSameQuiz() {
    setCurrentIndex(0);
    setSelectedAnswers({});
    setFinished(false);
  }

  function reviewAttempt(attempt) {
    setQuestions(attempt.questions);
    setSelectedAnswers(attempt.selectedAnswers);
    setDifficulty(attempt.difficulty);
    setQuestionCount(attempt.questionCount);
    setCurrentIndex(0);
    setFinished(true);
    setShowHistory(false);
    setError("");
  }

  function deleteAttempt(attemptId) {
    setHistory((previous) =>
      previous.filter((attempt) => attempt.id !== attemptId)
    );
  }

  function clearHistory() {
    const confirmed = window.confirm(
      "Delete all saved quiz attempts? This cannot be undone."
    );

    if (confirmed) {
      setHistory([]);
    }
  }

  function returnToSetup() {
    setQuestions([]);
    setCurrentIndex(0);
    setSelectedAnswers({});
    setFinished(false);
    setError("");
    setShowHistory(false);
  }

  const currentQuestion = questions[currentIndex];
  const selectedAnswer = selectedAnswers[currentIndex];

  const score = questions.reduce(
    (total, question, index) =>
      total +
      (
        selectedAnswers[index] === question.correct_answer
          ? 1
          : 0
      ),
    0
  );

  return (
    <section
      className="quiz-card"
      aria-labelledby="quiz-heading"
    >
      {/* Header */}
      <div className="quiz-header">
        <div>
          <span className="quiz-eyebrow">
            ACTIVE RECALL
          </span>

          <h2 id="quiz-heading">
            AI Knowledge Check
          </h2>

          <p>
            Test what you learned from this video.
          </p>
        </div>

        <div className="quiz-header-actions">
          <button
            type="button"
            className="quiz-history-toggle"
            onClick={() => {
              setShowHistory((previous) => !previous);
              setError("");
            }}
          >
            {showHistory
              ? "← Back to quiz"
              : `📚 History (${history.length})`}
          </button>
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div className="quiz-error" role="alert">
          {error}
        </div>
      )}

      {/* Quiz history */}
      {showHistory && (
        <div className="quiz-history">
          <div className="quiz-history-heading">
            <div>
              <span className="quiz-eyebrow">
                YOUR LEARNING JOURNEY
              </span>

              <h3>Quiz History</h3>

              <p>
                Review previous attempts and track your progress.
              </p>
            </div>

            {history.length > 0 && (
              <button
                type="button"
                className="quiz-history-clear"
                onClick={clearHistory}
              >
                Clear history
              </button>
            )}
          </div>

          {history.length === 0 ? (
            <div className="quiz-history-empty">
              <span>📚</span>

              <h4>No quiz attempts yet</h4>

              <p>
                Complete your first quiz and your results
                will appear here.
              </p>

              <button
                type="button"
                className="quiz-primary-button"
                onClick={() => setShowHistory(false)}
              >
                Start learning
              </button>
            </div>
          ) : (
            <div className="quiz-history-list">
              {history.map((attempt) => (
                <article
                  className="quiz-history-item"
                  key={attempt.id}
                >
                  <div className="quiz-history-item-top">
                    <div className="quiz-history-video">
                      <span>▶</span>

                      <div>
                        <strong>
                          YouTube video
                        </strong>

                        <small>
                          {attempt.videoId}
                        </small>
                      </div>
                    </div>

                    <span
                      className={`quiz-history-score ${
                        attempt.percentage >= 80
                          ? "score-high"
                          : attempt.percentage >= 50
                            ? "score-medium"
                            : "score-low"
                      }`}
                    >
                      {attempt.percentage}%
                    </span>
                  </div>

                  <div className="quiz-history-meta">
                    <span>
                      📝 {attempt.questionCount} questions
                    </span>

                    <span>
                      🎯 {attempt.difficulty}
                    </span>

                    <span>
                      ✓ {attempt.score}/{attempt.questionCount}
                    </span>
                  </div>

                  <p className="quiz-history-date">
                    {formatDate(attempt.completedAt)}
                  </p>

                  <div className="quiz-history-actions">
                    <button
                      type="button"
                      className="quiz-primary-button"
                      onClick={() => reviewAttempt(attempt)}
                    >
                      Review attempt
                    </button>

                    <button
                      type="button"
                      className="quiz-delete-button"
                      onClick={() => deleteAttempt(attempt.id)}
                      aria-label="Delete quiz attempt"
                    >
                      🗑 Delete
                    </button>
                  </div>
                </article>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Quiz setup */}
      {!showHistory && questions.length === 0 && (
        <div className="quiz-setup">
          <div className="quiz-setup-intro">
            <span className="quiz-setup-icon">
              🧠
            </span>

            <div>
              <h3>Build your quiz</h3>

              <p>
                Questions are generated from the selected
                video's indexed transcript.
              </p>
            </div>
          </div>

          <div className="quiz-controls">
            <label>
              <span>Number of questions</span>

              <select
                value={questionCount}
                onChange={(event) =>
                  setQuestionCount(
                    Number(event.target.value)
                  )
                }
                disabled={loading}
              >
                <option value={5}>5 questions</option>
                <option value={10}>10 questions</option>
                <option value={15}>15 questions</option>
              </select>
            </label>

            <label>
              <span>Difficulty</span>

              <select
                value={difficulty}
                onChange={(event) =>
                  setDifficulty(event.target.value)
                }
                disabled={loading}
              >
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </select>
            </label>
          </div>

          <button
            type="button"
            className="quiz-primary-button quiz-generate-button"
            onClick={generateQuiz}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="quiz-spinner" />
                Generating quiz…
              </>
            ) : (
              "✨ Generate Quiz"
            )}
          </button>

          {loading && (
            <p className="quiz-loading-note">
              Your local AI is writing and reviewing the
              questions. Keep this page open.
            </p>
          )}
        </div>
      )}

      {/* Active question */}
      {!showHistory &&
        questions.length > 0 &&
        !finished &&
        currentQuestion && (
          <div className="quiz-question-panel">
            <div className="quiz-progress-track">
              <span
                style={{
                  width: `${
                    ((currentIndex + 1) / questions.length) * 100
                  }%`,
                }}
              />
            </div>

            <div className="quiz-question-topline">
              <span className="quiz-question-number">
                QUESTION{" "}
                {String(currentIndex + 1).padStart(2, "0")}
              </span>

              <span
                className={`quiz-difficulty difficulty-${difficulty}`}
              >
                {difficulty}
              </span>
            </div>

            <h3 className="quiz-question-text">
              {currentQuestion.question}
            </h3>

            <div
              className="quiz-options"
              role="group"
              aria-label="Answer options"
            >
              {currentQuestion.options.map(
                (option, optionIndex) => {
                  const answered =
                    selectedAnswer !== undefined;

                  const isCorrect =
                    optionIndex ===
                    currentQuestion.correct_answer;

                  const isSelected =
                    optionIndex === selectedAnswer;

                  let optionClass = "";

                  if (answered && isCorrect) {
                    optionClass = "option-correct";
                  } else if (
                    answered &&
                    isSelected &&
                    !isCorrect
                  ) {
                    optionClass = "option-incorrect";
                  } else if (isSelected) {
                    optionClass = "option-selected";
                  }

                  return (
                    <button
                      key={`${currentIndex}-${optionIndex}`}
                      type="button"
                      className={`quiz-option ${optionClass}`}
                      onClick={() =>
                        selectAnswer(optionIndex)
                      }
                      disabled={answered}
                    >
                      <span className="quiz-option-letter">
                        {String.fromCharCode(
                          65 + optionIndex
                        )}
                      </span>

                      <span className="quiz-option-text">
                        {option}
                      </span>

                      {answered && isCorrect && (
                        <span className="quiz-option-mark">
                          ✓
                        </span>
                      )}

                      {answered &&
                        isSelected &&
                        !isCorrect && (
                          <span className="quiz-option-mark">
                            ×
                          </span>
                        )}
                    </button>
                  );
                }
              )}
            </div>

            {selectedAnswer !== undefined && (
              <div
                className={`quiz-feedback ${
                  selectedAnswer ===
                  currentQuestion.correct_answer
                    ? "feedback-correct"
                    : "feedback-incorrect"
                }`}
                aria-live="polite"
              >
                <strong>
                  {selectedAnswer ===
                  currentQuestion.correct_answer
                    ? "Correct answer!"
                    : "Not quite — review the explanation."}
                </strong>

                <p>
                  {currentQuestion.explanation}
                </p>

                <button
                  type="button"
                  className="quiz-timestamp-button"
                  onClick={() =>
                    onJumpToTimestamp?.(
                      currentQuestion.timestamp_seconds
                    )
                  }
                >
                  ▶ Watch this part of the video

                  <span>
                    {Math.floor(
                      currentQuestion.timestamp_seconds / 60
                    )}
                    :
                    {String(
                      currentQuestion.timestamp_seconds % 60
                    ).padStart(2, "0")}
                  </span>
                </button>
              </div>
            )}

            <div className="quiz-question-footer">
              <span>
                {Object.keys(selectedAnswers).length} answered
              </span>

              <button
                type="button"
                className="quiz-primary-button quiz-next-button"
                onClick={nextQuestion}
                disabled={selectedAnswer === undefined}
              >
                {currentIndex === questions.length - 1
                  ? "View results"
                  : "Next question →"}
              </button>
            </div>
          </div>
        )}

      {/* Results */}
      {!showHistory && finished && (
        <div className="quiz-results">
          <div className="quiz-results-icon">
            {score / questions.length >= 0.8
              ? "🏆"
              : "🎯"}
          </div>

          <span className="quiz-eyebrow">
            QUIZ COMPLETE
          </span>

          <h3>
            {score / questions.length >= 0.8
              ? "Excellent work!"
              : score / questions.length >= 0.5
                ? "Good effort!"
                : "Keep learning!"}
          </h3>

          <div className="quiz-score">
            <strong>{score}</strong>

            <span>
              / {questions.length}
            </span>
          </div>

          <p>
            You answered {score} of {questions.length}{" "}
            questions correctly (
            {Math.round(
              (score / questions.length) * 100
            )}
            %).
          </p>

          <div className="quiz-results-actions">
            <button
              type="button"
              className="quiz-secondary-button"
              onClick={restartSameQuiz}
            >
              Retry this quiz
            </button>

            <button
              type="button"
              className="quiz-secondary-button"
              onClick={() => setShowHistory(true)}
            >
              📚 Quiz history
            </button>

            <button
              type="button"
              className="quiz-primary-button"
              onClick={returnToSetup}
            >
              New quiz
            </button>
          </div>

          <div className="quiz-answer-review">
            <h4>Answer review</h4>

            {questions.map((question, index) => {
              const correct =
                selectedAnswers[index] ===
                question.correct_answer;

              return (
                <div
                  className="quiz-review-item"
                  key={`review-${index}`}
                >
                  <span
                    className={
                      correct
                        ? "review-check"
                        : "review-cross"
                    }
                  >
                    {correct ? "✓" : "×"}
                  </span>

                  <div>
                    <strong>
                      {index + 1}. {question.question}
                    </strong>

                    <p>
                      Correct answer:{" "}
                      {
                        question.options[
                          question.correct_answer
                        ]
                      }
                    </p>

                    <button
                      type="button"
                      className="quiz-review-timestamp"
                      onClick={() =>
                        onJumpToTimestamp?.(
                          question.timestamp_seconds
                        )
                      }
                    >
                      ▶ Jump to{" "}
                      {Math.floor(
                        question.timestamp_seconds / 60
                      )}
                      :
                      {String(
                        question.timestamp_seconds % 60
                      ).padStart(2, "0")}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}