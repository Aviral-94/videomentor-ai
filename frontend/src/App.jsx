import { useEffect, useState } from "react"
import YouTube from "react-youtube"
import "./App.css"

function App() {
  const [youtubeUrl, setYoutubeUrl] = useState("")
  const [videoId, setVideoId] = useState("")
  const [player, setPlayer] = useState(null)

  const [question, setQuestion] = useState("")
  const [answer, setAnswer] = useState("")
  const [timestamp, setTimestamp] = useState("")
  const [timestampSeconds, setTimestampSeconds] = useState(null)

  const [loading, setLoading] = useState(false)

  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem("videomentor-theme") === "dark"
  })

  useEffect(() => {
    localStorage.setItem(
      "videomentor-theme",
      darkMode ? "dark" : "light"
    )
  }, [darkMode])


  // ============================================
  // Extract YouTube ID
  // ============================================

  function extractVideoId(url) {
    try {
      const parsedUrl = new URL(url)

      if (parsedUrl.hostname.includes("youtube.com")) {
        return parsedUrl.searchParams.get("v")
      }

      if (parsedUrl.hostname.includes("youtu.be")) {
        return parsedUrl.pathname.slice(1)
      }

      return null
    } catch {
      return null
    }
  }


  // ============================================
  // Load Video
  // ============================================

  function loadVideo() {
    const id = extractVideoId(youtubeUrl)

    if (!id) {
      alert("Please enter a valid YouTube URL")
      return
    }

    setVideoId(id)

    setAnswer("")
    setTimestamp("")
    setTimestampSeconds(null)

    setPlayer(null)
  }


  // ============================================
  // YouTube Player Ready
  // ============================================

  function onPlayerReady(event) {
    setPlayer(event.target)
  }


  // ============================================
  // Ask Question
  // ============================================

  async function askQuestion() {
    if (!question.trim()) {
      return
    }

    if (!videoId) {
      alert("Please load a YouTube video first.")
      return
    }

    if (!player) {
      alert("Video player is not ready yet.")
      return
    }

    const currentTime = player.getCurrentTime()

    setLoading(true)

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/api/chat",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json"
          },

          body: JSON.stringify({
            video_id: videoId,
            current_time: currentTime,
            question: question
          })
        }
      )

      if (!response.ok) {
        throw new Error("API request failed")
      }

      const data = await response.json()

      console.log("VideoMentor response:", data)

      setAnswer(data.answer)

      setTimestamp(data.timestamp)

      setTimestampSeconds(
        data.timestamp_seconds
      )

    } catch (error) {
      console.error(error)

      setAnswer(
        "Something went wrong while contacting VideoMentor."
      )

      setTimestamp("")
      setTimestampSeconds(null)

    } finally {
      setLoading(false)
    }
  }


  // ============================================
  // Jump to Relevant Moment
  // ============================================

  function jumpToTimestamp() {
    if (!player) {
      console.log("YouTube player not ready")
      return
    }

    if (
      timestampSeconds === null ||
      timestampSeconds === undefined
    ) {
      console.log("No timestamp available")
      return
    }

    console.log(
      "Jumping to:",
      timestampSeconds,
      "seconds"
    )

    player.seekTo(
      Number(timestampSeconds),
      true
    )

    player.playVideo()
  }


  // ============================================
  // Enter Key
  // ============================================

  function handleQuestionKeyDown(event) {
    if (event.key === "Enter") {
      event.preventDefault()
      askQuestion()
    }
  }


  // ============================================
  // UI
  // ============================================

  return (
    <div
      className={
        darkMode
          ? "app dark"
          : "app"
      }
    >

      {/* ========================================
          HEADER
      ======================================== */}

      <header className="header">

        <div className="brand">

          <div className="logo">
            🎓
          </div>

          <div>

            <h1>
              VideoMentor AI
            </h1>

            <span>
              Your personal YouTube learning assistant
            </span>

          </div>

        </div>


        <button
          className="theme-toggle"
          onClick={() =>
            setDarkMode(!darkMode)
          }
        >
          {darkMode
            ? "☀️ Light"
            : "🌙 Dark"}
        </button>

      </header>


      {/* ========================================
          URL BAR
      ======================================== */}

      <section className="url-section">

        <input
          type="text"
          placeholder="Paste a YouTube video URL..."
          value={youtubeUrl}
          onChange={(event) =>
            setYoutubeUrl(
              event.target.value
            )
          }
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              loadVideo()
            }
          }}
        />

        <button
          className="primary-button"
          onClick={loadVideo}
        >
          Load Video
        </button>

      </section>


      {/* ========================================
          MAIN
      ======================================== */}

      <main className="main-layout">


        {/* ======================================
            VIDEO
        ====================================== */}

        <section className="video-section">

          <div className="section-title">

            <span>
              🎥
            </span>

            <h2>
              Video
            </h2>

          </div>


          <div className="video-card">

            {videoId ? (

              <YouTube
                videoId={videoId}

                className="youtube-player"

                opts={{
                  width: "100%",
                  height: "100%",

                  playerVars: {
                    autoplay: 0,
                    rel: 0
                  }
                }}

                onReady={onPlayerReady}
              />

            ) : (

              <div className="video-placeholder">

                <div className="placeholder-icon">
                  ▶
                </div>

                <h3>
                  Start learning
                </h3>

                <p>
                  Paste a YouTube URL above to begin.
                </p>

              </div>

            )}

          </div>

        </section>


        {/* ======================================
            AI TUTOR
        ====================================== */}

        <section className="tutor-section">

          <div className="section-title">

            <span>
              🤖
            </span>

            <h2>
              AI Tutor
            </h2>

          </div>


          <div className="chat-card">


            {/* ==================================
                WELCOME
            ================================== */}

            {!answer && !loading && (

              <div className="welcome">

                <div className="bot-icon">
                  🤖
                </div>

                <h3>
                  Ask me anything about the video
                </h3>

                <p>
                  I'll explain concepts using
                  the video you're watching.
                </p>

              </div>

            )}


            {/* ==================================
                LOADING
            ================================== */}

            {loading && (

              <div className="loading">

                <div className="loader"></div>

                <span>
                  VideoMentor is thinking...
                </span>

              </div>

            )}


            {/* ==================================
                ANSWER
            ================================== */}

            {answer && !loading && (

              <div className="answer">

                <div className="answer-header">

                  <span>
                    🤖
                  </span>

                  <strong>
                    VideoMentor
                  </strong>

                </div>


                <p>
                  {answer}
                </p>


                {timestamp && (

                  <div className="timestamp">

                    <span>
                      📍 Relevant moment:{" "}
                      <strong>
                        {timestamp}
                      </strong>
                    </span>


                    <button
                      onClick={
                        jumpToTimestamp
                      }
                    >
                      ▶ Jump to moment
                    </button>

                  </div>

                )}

              </div>

            )}


            {/* ==================================
                QUESTION
            ================================== */}

            <div className="question-box">

              <input
                type="text"
                placeholder="Ask anything about this video..."
                value={question}
                onChange={(event) =>
                  setQuestion(
                    event.target.value
                  )
                }
                onKeyDown={
                  handleQuestionKeyDown
                }
              />

              <button
                onClick={askQuestion}
                disabled={loading}
              >
                {loading
                  ? "..."
                  : "➤"}
              </button>

            </div>

          </div>

        </section>

      </main>

    </div>
  )
}

export default App