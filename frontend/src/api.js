/**
 * VideoMentor AI Frontend API Client
 *
 * Implements request timeouts with AbortController, timer cleanup, and
 * comprehensive error classification across 7 distinct situations:
 * 1. Request timeout (45s)
 * 2. Backend unavailable
 * 3. Network connection failure
 * 4. HTTP 400 / 404 client errors
 * 5. HTTP 422 validation errors
 * 6. HTTP 500 / 5xx server errors
 * 7. Unexpected response format
 */

export const API_URL =
  (typeof globalThis !== "undefined" && globalThis.process?.env?.VITE_API_URL) ||
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_URL) ||
  "http://127.0.0.1:8001";

export const TIMEOUTS = {
  CHAT: 45000,
  NOTES: 45000,
  STUDY_GUIDE: 45000,
  QUIZ: 45000,
  INGEST: 45000,
  DELETE_HISTORY: 15000,
  DEFAULT: 45000,
};

export class ApiError extends Error {
  constructor(message, { type = "UNKNOWN", status = null, details = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.type = type;
    this.status = status;
    this.details = details;
  }
}

/**
 * Extracts error details from a FastAPI / Pydantic JSON response.
 */
function extractErrorDetail(data) {
  if (!data) return "";
  if (typeof data.detail === "string") {
    return data.detail;
  }
  if (Array.isArray(data.detail)) {
    return data.detail
      .map((item) => {
        if (!item || typeof item !== "object") return String(item);
        const loc = Array.isArray(item.loc)
          ? item.loc.filter((part) => part !== "body").join(".")
          : "";
        const msg = item.msg || item.message || "Invalid value";
        return loc ? `${loc}: ${msg}` : msg;
      })
      .join("; ");
  }
  if (typeof data.message === "string") {
    return data.message;
  }
  return "";
}

/**
 * Fetch wrapper with AbortController timeout and robust error classification.
 *
 * @param {string} url - Target URL.
 * @param {RequestInit} [options={}] - Standard fetch options.
 * @param {number} [timeoutMs=TIMEOUTS.DEFAULT] - Timeout in milliseconds.
 * @returns {Promise<any>} Parsed JSON response.
 */
export async function fetchWithTimeout(url, options = {}, timeoutMs = TIMEOUTS.DEFAULT) {
  const controller = new AbortController();
  let isTimeout = false;

  const timerId = setTimeout(() => {
    isTimeout = true;
    controller.abort();
  }, timeoutMs);

  let onExternalAbort = null;
  if (options.signal) {
    if (options.signal.aborted) {
      clearTimeout(timerId);
      controller.abort(options.signal.reason);
    } else {
      onExternalAbort = () => {
        clearTimeout(timerId);
        controller.abort(options.signal.reason);
      };
      options.signal.addEventListener("abort", onExternalAbort, { once: true });
    }
  }

  let response;
  try {
    response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
  } catch (error) {
    // 1. Request timeout
    if (isTimeout) {
      throw new ApiError(
        "The AI is taking longer than expected. Please try again.",
        { type: "TIMEOUT", details: error.message }
      );
    }

    // Explicit caller cancellation
    if (error.name === "AbortError") {
      throw new ApiError("The request was cancelled.", {
        type: "ABORTED",
        details: error.message,
      });
    }

    if (error instanceof ApiError) {
      throw error;
    }

    // 2. Backend unavailable / 3. Network connection failure
    throw new ApiError(
      "Unable to connect to VideoMentor AI server. Please check whether the backend is running.",
      { type: "CONNECTION", details: error.message }
    );
  } finally {
    clearTimeout(timerId);
    if (options.signal && onExternalAbort) {
      options.signal.removeEventListener("abort", onExternalAbort);
    }
  }

  // Handle non-2xx responses
  if (!response.ok) {
    let errorDetail;
    try {
      const errorJson = await response.json();
      errorDetail = extractErrorDetail(errorJson);
    } catch {
      try {
        errorDetail = (await response.text()).trim();
      } catch {
        errorDetail = "";
      }
    }

    // 4. HTTP 400 / 404 errors
    if (response.status === 400) {
      throw new ApiError(
        errorDetail || "Invalid request (HTTP 400). Please check your input and try again.",
        { type: "BAD_REQUEST", status: 400, details: errorDetail }
      );
    }

    if (response.status === 404) {
      throw new ApiError(
        errorDetail || "The requested resource was not found (HTTP 404).",
        { type: "NOT_FOUND", status: 404, details: errorDetail }
      );
    }

    // 5. HTTP 422 validation errors
    if (response.status === 422) {
      throw new ApiError(
        errorDetail
          ? `Validation error (HTTP 422): ${errorDetail}`
          : "Invalid request format or parameters (HTTP 422).",
        { type: "VALIDATION", status: 422, details: errorDetail }
      );
    }

    // 6. HTTP 500 / 5xx server errors
    if (response.status === 500) {
      throw new ApiError(
        errorDetail || "Internal server error (HTTP 500). Please check the backend terminal and try again.",
        { type: "SERVER_ERROR", status: 500, details: errorDetail }
      );
    }

    if (response.status >= 501 && response.status <= 599) {
      throw new ApiError(
        errorDetail || `Server error (HTTP ${response.status}). VideoMentor AI server or Ollama may be temporarily unavailable.`,
        { type: "SERVER_ERROR", status: response.status, details: errorDetail }
      );
    }

    throw new ApiError(
      errorDetail || `Request failed with HTTP status ${response.status}.`,
      { type: "HTTP_ERROR", status: response.status, details: errorDetail }
    );
  }

  // 7. Unexpected response format (JSON parsing)
  let data;
  try {
    data = await response.json();
  } catch (parseError) {
    throw new ApiError(
      "Received an unexpected response format from the server. Expected JSON.",
      { type: "PARSE_ERROR", details: parseError.message }
    );
  }

  return data;
}

/**
 * Ask a question across selected video(s) (Chat request).
 * Recommended timeout: 45 seconds.
 */
export async function sendChatMessage(payload, options = {}) {
  const data = await fetchWithTimeout(
    `${API_URL}/api/chat`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: payload.conversation_id,
        video_id: payload.video_id,
        video_ids: payload.video_ids,
        current_time: payload.current_time,
        question: payload.question,
        mode: payload.mode || "explain",
      }),
      signal: options.signal,
    },
    TIMEOUTS.CHAT
  );

  if (!data || typeof data !== "object") {
    throw new ApiError(
      "The server returned an unexpected chat response format.",
      { type: "UNEXPECTED_FORMAT" }
    );
  }

  return data;
}

/**
 * Ingest/Index a video transcript.
 * Timeout: 45 seconds.
 */
export async function ingestVideo(url, options = {}) {
  const data = await fetchWithTimeout(
    `${API_URL}/api/videos/ingest`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
      signal: options.signal,
    },
    TIMEOUTS.INGEST
  );

  if (!data || typeof data !== "object") {
    throw new ApiError(
      "The server returned an unexpected video ingestion response.",
      { type: "UNEXPECTED_FORMAT" }
    );
  }

  return data;
}

/**
 * Clear server-side conversation memory.
 * Timeout: 15 seconds.
 */
export async function clearChatHistory(conversationId, options = {}) {
  return await fetchWithTimeout(
    `${API_URL}/api/chat/history/${encodeURIComponent(conversationId)}`,
    {
      method: "DELETE",
      signal: options.signal,
    },
    TIMEOUTS.DELETE_HISTORY
  );
}

/**
 * Generate AI Quiz for a video.
 * Recommended timeout: 45 seconds.
 */
export async function generateQuiz({ videoId, numQuestions = 5, difficulty = "medium" }, options = {}) {
  const data = await fetchWithTimeout(
    `${API_URL}/api/quiz`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_id: videoId,
        num_questions: numQuestions,
        difficulty,
      }),
      signal: options.signal,
    },
    TIMEOUTS.QUIZ
  );

  if (!data || !Array.isArray(data.questions) || data.questions.length === 0) {
    throw new ApiError(
      "The server returned an unexpected quiz response format or no questions.",
      { type: "UNEXPECTED_FORMAT" }
    );
  }

  return data;
}

/**
 * Generate Visual Study Guide (Notes & Flowchart) for a video.
 * Recommended timeout: 45 seconds.
 */
export async function generateStudyGuide({ videoId }, options = {}) {
  const data = await fetchWithTimeout(
    `${API_URL}/api/study-guide`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId }),
      signal: options.signal,
    },
    TIMEOUTS.STUDY_GUIDE
  );

  if (!data || typeof data.notes !== "string" || typeof data.flowchart !== "string") {
    throw new ApiError(
      "The server returned incomplete study-guide data (missing notes or flowchart).",
      { type: "UNEXPECTED_FORMAT" }
    );
  }

  return data;
}

/**
 * Generate Video Notes for a video.
 * Recommended timeout: 45 seconds.
 */
export async function generateNotes({ videoId }, options = {}) {
  const data = await fetchWithTimeout(
    `${API_URL}/api/notes`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId }),
      signal: options.signal,
    },
    TIMEOUTS.NOTES
  );

  if (!data || typeof data.notes !== "string") {
    throw new ApiError(
      "The server returned incomplete notes data.",
      { type: "UNEXPECTED_FORMAT" }
    );
  }

  return data;
}
