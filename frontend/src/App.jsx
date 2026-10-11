import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import YouTube from "react-youtube";
import StudyGuide from "./StudyGuide";
import Quiz from "./Quiz";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function createConversationId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }

  return `conversation-${Date.now()}-${Math.random()
    .toString(36)
    .slice(2)}`;
}

function formatMessageTime(date = new Date()) {
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function App() {
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [videoId, setVideoId] = useState("");
  const [player, setPlayer] = useState(null);

  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [tutorMode, setTutorMode] = useState("explain");

  // Visual Study Guide state
  const [showStudyGuide, setShowStudyGuide] = useState(false);
  const [showQuiz, setShowQuiz] = useState(false);

  const [conversationId, setConversationId] = useState(() => {
    return (
      localStorage.getItem("videomentor-conversation-id") ||
      createConversationId()
    );
  });

  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem("videomentor-theme") === "dark";
  });

  const messagesEndRef = useRef(null);

  // Persist the conversation ID.
  useEffect(() => {
    localStorage.setItem(
      "videomentor-conversation-id",
      conversationId
    );
  }, [conversationId]);

  // Persist the selected theme.
  useEffect(() => {
    localStorage.setItem(
      "videomentor-theme",
      darkMode ? "dark" : "light"
    );
  }, [darkMode]);

  // Automatically scroll to the latest message.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [messages, loading]);

  // Extract YouTube video ID.
  function extractVideoId(url) {
    try {
      const parsedUrl = new URL(url);

      if (parsedUrl.hostname.includes("youtu.be")) {
        return (
          parsedUrl.pathname.split("/").filter(Boolean)[0] || null
        );
      }

      if (
        parsedUrl.hostname.includes("youtube.com") ||
        parsedUrl.hostname.includes("youtube-nocookie.com")
      ) {
        if (parsedUrl.pathname === "/watch") {
          return parsedUrl.searchParams.get("v");
        }

        const pathParts = parsedUrl.pathname
          .split("/")
          .filter(Boolean);

        if (
          ["embed", "shorts", "live"].includes(pathParts[0]) &&
          pathParts[1]
        ) {
          return pathParts[1];
        }
      }

      return null;
    } catch {
      return null;
    }
  }

  // Start a new conversation.
  async function startNewChat() {
    if (loading) return;

    const oldConversationId = conversationId;

    try {
      await fetch(
        `${API_URL}/api/chat/history/${encodeURIComponent(
          oldConversationId
        )}`,
        {
          method: "DELETE",
        }
      );
    } catch (error) {
      console.warn("Could not clear server memory:", error);
    }

    setConversationId(createConversationId());
    setMessages([]);
    setQuestion("");
  }

  // Load a YouTube video.
  function loadVideo() {
    const extractedId = extractVideoId(youtubeUrl.trim());

    if (!extractedId) {
      alert("Please enter a valid YouTube video URL.");
      return;
    }

    if (extractedId !== videoId) {
      // Start a fresh conversation when switching videos.
      setConversationId(createConversationId());
      setMessages([]);

      // Reset video-specific tools when switching videos.
      setShowStudyGuide(false);
      setShowQuiz(false);
    }

    setVideoId(extractedId);
    setPlayer(null);
  }

  // Called when the YouTube player is ready.
  function onPlayerReady(event) {
    setPlayer(event.target);
  }

  // Ask the AI tutor a question.
  async function askQuestion(event) {
    if (event) {
      event.preventDefault();
    }

    const trimmedQuestion = question.trim();

    if (!trimmedQuestion) return;

    if (!videoId) {
      alert("Please load a YouTube video first.");
      return;
    }

    if (!player) {
      alert("Please wait for the YouTube player to load.");
      return;
    }

    if (loading) return;

    const currentTime = player.getCurrentTime();
    const submittedQuestion = trimmedQuestion;
    const activeConversationId = conversationId;

    const userMessage = {
      id: createConversationId(),
      role: "user",
      content: submittedQuestion,
      createdAt: formatMessageTime(),
    };

    // Show the user's message immediately.
    setMessages((previousMessages) => [
      ...previousMessages,
      userMessage,
    ]);

    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          conversation_id: activeConversationId,
          video_id: videoId,
          current_time: currentTime,
          question: submittedQuestion,
          mode: tutorMode,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();

        throw new Error(
          `Server returned ${response.status}: ${errorText}`
        );
      }

      const data = await response.json();

      const assistantMessage = {
        id: createConversationId(),
        role: "assistant",
        content: data.answer || "I could not generate an answer.",
        timestamp: data.timestamp || null,
        timestampSeconds:
          typeof data.timestamp_seconds === "number"
            ? data.timestamp_seconds
            : null,
        youtubeUrl: data.youtube_url || null,
        createdAt: formatMessageTime(),
      };

      // Avoid placing an old response into a newly started chat.
      if (activeConversationId === conversationId) {
        setMessages((previousMessages) => [
          ...previousMessages,
          assistantMessage,
        ]);
      }
    } catch (error) {
      console.error("Chat request failed:", error);

      setMessages((previousMessages) => [
        ...previousMessages,
        {
          id: createConversationId(),
          role: "assistant",
          content:
            "Sorry, I could not connect to the AI tutor. " +
            "Please check that the FastAPI backend, Ollama, " +
            "and Qdrant are running, then try again.\n\n" +
            `Details: ${error.message}`,
          createdAt: formatMessageTime(),
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  // Jump to a timestamp in the YouTube video.
  function jumpToTimestamp(seconds) {
    if (!player || typeof seconds !== "number") {
      return;
    }

    player.seekTo(seconds, true);
    player.playVideo();
  }

  // Enter submits; Shift+Enter creates a new line.
  function handleQuestionKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      askQuestion(event);
    }
  }

  const youtubeOptions = {
    width: "100%",
    playerVars: {
      autoplay: 0,
      modestbranding: 1,
      rel: 0,
    },
  };

  return (
    <div className={`app ${darkMode ? "dark-mode" : ""}`}>
      {/* Header */}
      <header className="header">
        <div className="brand">
          <div className="logo">▶</div>

          <div>
            <h1>VideoMentor AI</h1>
            <p>Your personal AI learning companion</p>
          </div>
        </div>

        <div className="header-actions">
          <button
            type="button"
            className="new-chat-button"
            onClick={startNewChat}
            disabled={loading}
          >
            + New Chat
          </button>

          <button
            type="button"
            className="theme-toggle"
            onClick={() =>
              setDarkMode((previous) => !previous)
            }
            aria-label="Toggle dark mode"
            title="Toggle theme"
          >
            {darkMode ? "☀️" : "🌙"}
          </button>
        </div>
      </header>

      {/* YouTube URL input */}
      <section className="url-section">
        <div className="url-input-wrapper">
          <span className="url-icon">🔗</span>

          <input
            type="text"
            value={youtubeUrl}
            onChange={(event) =>
              setYoutubeUrl(event.target.value)
            }
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                loadVideo();
              }
            }}
            placeholder="Paste a YouTube video URL..."
            aria-label="YouTube video URL"
          />

          <button type="button" onClick={loadVideo}>
            Load Video
          </button>
        </div>
      </section>

      {/* Main layout */}
      <main className="main-layout">
        {/* Video panel */}
        <section className="video-card">
          <div className="card-heading">
            <div>
              <h2>Video Player</h2>
              <p>Watch and learn at your own pace</p>
            </div>

            <span className="status-badge">
              {videoId ? "Video Loaded" : "Ready"}
            </span>
          </div>

          <div className="video-container">
            {videoId ? (
              <YouTube
                key={videoId}
                videoId={videoId}
                opts={youtubeOptions}
                onReady={onPlayerReady}
                className="youtube-player"
                iframeClassName="youtube-iframe"
              />
            ) : (
              <div className="video-placeholder">
                <div className="placeholder-icon">▶</div>

                <h3>Your learning journey starts here</h3>

                <p>
                  Paste a YouTube URL above to start learning
                  with your AI tutor.
                </p>
              </div>
            )}
          </div>

          {videoId && (
            <>
              <div className="video-info">
                <span className="video-info-dot" />
                <span>Ask questions about this video</span>
              </div>

              {/* Visual Study Guide button */}
              <button
                type="button"
                className="study-guide-open-button"
                onClick={() =>
                  setShowStudyGuide((previous) => !previous)
                }
                aria-expanded={showStudyGuide}
              >
                {showStudyGuide
                  ? "✕ Hide Study Guide"
                  : "✨ Generate Visual Study Guide"}
              </button>

              <button
                type="button"
                className="quiz-open-button"
                onClick={() => setShowQuiz((previous) => !previous)}
                aria-expanded={showQuiz}
              >
                {showQuiz ? "✕ Hide Quiz" : "🧠 Generate AI Quiz"}
              </button>
            </>
          )}
        </section>

        {/* AI Tutor panel */}
        <section className="chat-card">
          <div className="card-heading conversation-header">
            <div>
              <h2>AI Tutor</h2>

              <p>
                Ask questions and explore concepts ·{" "}
                {tutorMode[0].toUpperCase() + tutorMode.slice(1)} mode
              </p>
            </div>

            <span className="status-badge">
              {loading ? "Thinking..." : "AI Ready"}
            </span>
          </div>

          {/* Tutor mode selector */}
          <div
            className="tutor-mode-section"
            aria-label="Choose tutor mode"
          >
            <span className="tutor-mode-label">
              Tutor mode
            </span>

            <div
              className="tutor-mode-options"
              role="group"
              aria-label="Tutor modes"
            >
              {[
                {
                  id: "explain",
                  icon: "📘",
                  label: "Explain",
                },
                {
                  id: "summarize",
                  icon: "📝",
                  label: "Summarize",
                },
                {
                  id: "quiz",
                  icon: "🧠",
                  label: "Quiz",
                },
                {
                  id: "interview",
                  icon: "🎤",
                  label: "Interview",
                },
              ].map((mode) => (
                <button
                  key={mode.id}
                  type="button"
                  className={`tutor-mode-button ${
                    tutorMode === mode.id ? "active" : ""
                  }`}
                  onClick={() => setTutorMode(mode.id)}
                  aria-pressed={tutorMode === mode.id}
                  disabled={loading}
                  title={`Switch to ${mode.label} mode`}
                >
                  <span aria-hidden="true">
                    {mode.icon}
                  </span>
                  <span>{mode.label}</span>
                </button>
              ))}
            </div>

            <p className="tutor-mode-description">
              {tutorMode === "explain" &&
                "Understand concepts step by step."}

              {tutorMode === "summarize" &&
                "Get the key points from the video."}

              {tutorMode === "quiz" &&
                "Answer questions to test your understanding."}

              {tutorMode === "interview" &&
                "Practice interview questions and get feedback."}
            </p>
          </div>

          {/* Conversation messages */}
          <div className="message-list">
            {messages.length === 0 && !loading && (
              <div className="welcome">
                <div className="welcome-icon">✨</div>

                <h3>What would you like to learn?</h3>

                <p>
                  Ask me to explain a concept, clarify a topic,
                  or answer a question about your video.
                </p>

                <div className="suggestion-list">
                  <button
                    type="button"
                    onClick={() =>
                      setQuestion(
                        "Explain the main concept of this video."
                      )
                    }
                  >
                    Explain the main concept
                  </button>

                  <button
                    type="button"
                    onClick={() =>
                      setQuestion(
                        "Summarize what has been explained so far."
                      )
                    }
                  >
                    Summarize this topic
                  </button>

                  <button
                    type="button"
                    onClick={() =>
                      setQuestion(
                        "Give me an example of this concept."
                      )
                    }
                  >
                    Give me an example
                  </button>
                </div>
              </div>
            )}

            {messages.map((message) => (
              <div
                key={message.id}
                className={`message-row ${
                  message.role === "user"
                    ? "user-message-row"
                    : "assistant-message-row"
                }`}
              >
                <div className="message-avatar">
                  {message.role === "user" ? "👤" : "✨"}
                </div>

                <div className="message-content">
                  <div className="message-meta">
                    <span>
                      {message.role === "user"
                        ? "You"
                        : "VideoMentor AI"}
                    </span>

                    <span>{message.createdAt}</span>
                  </div>

                  <div
                    className={`message-bubble ${
                      message.role === "user"
                        ? "user-bubble"
                        : "assistant-bubble"
                    } ${
                      message.isError ? "error-bubble" : ""
                    }`}
                  >
                    <div className="message-text">
                      <ReactMarkdown>
                       {message.content || ""}
                      </ReactMarkdown>
                    </div>

                    {message.role === "assistant" &&
                      message.timestampSeconds !== null &&
                      message.timestampSeconds !== undefined && (
                        <div className="message-timestamp">
                          <span>
                            📍{" "}
                            {message.timestamp ||
                              "Jump to context"}
                          </span>

                          <button
                            type="button"
                            onClick={() =>
                              jumpToTimestamp(
                                message.timestampSeconds
                              )
                            }
                            disabled={!player}
                          >
                            ▶ Watch context
                          </button>
                        </div>
                      )}
                  </div>
                </div>
              </div>
            ))}

            {loading && (
              <div className="message-row assistant-message-row">
                <div className="message-avatar">✨</div>

                <div className="message-content">
                  <div className="message-meta">
                    <span>VideoMentor AI</span>
                    <span>Thinking...</span>
                  </div>

                  <div className="message-bubble assistant-bubble">
                    <div className="thinking-indicator">
                      <span />
                      <span />
                      <span />
                    </div>

                    <p>Thinking about your question...</p>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Question input */}
          <form
            className="question-box"
            onSubmit={askQuestion}
          >
            <textarea
              value={question}
              onChange={(event) =>
                setQuestion(event.target.value)
              }
              onKeyDown={handleQuestionKeyDown}
              placeholder={
                videoId
                  ? tutorMode === "quiz"
                    ? "Answer the quiz question..."
                    : tutorMode === "interview"
                      ? "Type your interview answer..."
                      : tutorMode === "summarize"
                        ? "What would you like summarized?"
                        : "Ask anything about this video..."
                  : "Load a video to start asking questions..."
              }
              disabled={!videoId || loading}
              rows={2}
              aria-label="Ask the AI tutor"
            />

            <button
              type="submit"
              disabled={
                !videoId ||
                !player ||
                loading ||
                !question.trim()
              }
              aria-label="Send message"
              title="Send message"
            >
              {loading ? "..." : "➤"}
            </button>
          </form>

          <div className="chat-footer">
            <span>Powered by local AI</span>
            <span>Conversation memory enabled</span>
          </div>
        </section>
      </main>

      {/* Interactive AI Quiz */}
      {showQuiz && videoId && (
        <div className="quiz-wrapper">
          <Quiz
            videoId={videoId}
            onJumpToTimestamp={jumpToTimestamp}
          />
        </div>
      )}

      {/* Visual Study Guide */}
      {showStudyGuide && videoId && (
        <div className="study-guide-wrapper">
          <StudyGuide
            videoId={videoId}
            onJumpToTimestamp={jumpToTimestamp}
            onClose={() => setShowStudyGuide(false)}
          />
        </div>
      )}
    </div>
  );
}

export default App;