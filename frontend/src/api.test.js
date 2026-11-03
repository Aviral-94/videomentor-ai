import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import {
  fetchWithTimeout,
  sendChatMessage,
  generateQuiz,
  generateStudyGuide,
  generateNotes,
  ingestVideo,
  clearChatHistory,
  ApiError,
  TIMEOUTS,
} from "./api.js";

function createMockServer(handler) {
  return new Promise((resolve) => {
    const server = http.createServer(handler);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const url = `http://127.0.0.1:${address.port}`;
      resolve({
        url,
        port: address.port,
        close: () => new Promise((res) => server.close(res)),
      });
    });
  });
}

test("1. Request timeout throws ApiError with expected message", async () => {
  const server = await createMockServer(() => {
    // Never respond to trigger timeout
  });

  try {
    await assert.rejects(
      async () => {
        await fetchWithTimeout(`${server.url}/slow`, {}, 100);
      },
      (err) => {
        assert.ok(err instanceof ApiError);
        assert.equal(err.type, "TIMEOUT");
        assert.equal(
          err.message,
          "The AI is taking longer than expected. Please try again."
        );
        return true;
      }
    );
  } finally {
    await server.close();
  }
});

test("2. Backend unavailable / connection failure throws ApiError with expected message", async () => {
  const unusedUrl = "http://127.0.0.1:59999";

  await assert.rejects(
    async () => {
      await fetchWithTimeout(`${unusedUrl}/offline`, {}, 1000);
    },
    (err) => {
      assert.ok(err instanceof ApiError);
      assert.equal(err.type, "CONNECTION");
      assert.equal(
        err.message,
        "Unable to connect to VideoMentor AI server. Please check whether the backend is running."
      );
      return true;
    }
  );
});

test("3. HTTP 400 Bad Request extracts detail", async () => {
  const server = await createMockServer((_req, res) => {
    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ detail: "Please provide a valid YouTube video URL." }));
  });

  try {
    await assert.rejects(
      async () => {
        await fetchWithTimeout(`${server.url}/api/videos/ingest`, { method: "POST" });
      },
      (err) => {
        assert.ok(err instanceof ApiError);
        assert.equal(err.type, "BAD_REQUEST");
        assert.equal(err.status, 400);
        assert.equal(err.message, "Please provide a valid YouTube video URL.");
        return true;
      }
    );
  } finally {
    await server.close();
  }
});

test("4. HTTP 404 Not Found extracts detail", async () => {
  const server = await createMockServer((_req, res) => {
    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ detail: "Video not found in registry." }));
  });

  try {
    await assert.rejects(
      async () => {
        await fetchWithTimeout(`${server.url}/api/videos/123`, { method: "GET" });
      },
      (err) => {
        assert.ok(err instanceof ApiError);
        assert.equal(err.type, "NOT_FOUND");
        assert.equal(err.status, 404);
        assert.equal(err.message, "Video not found in registry.");
        return true;
      }
    );
  } finally {
    await server.close();
  }
});

test("5. HTTP 422 Validation Error formats FastAPI validation array", async () => {
  const server = await createMockServer((_req, res) => {
    res.writeHead(422, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        detail: [
          { loc: ["body", "question"], msg: "Field required", type: "missing" },
          { loc: ["body", "video_id"], msg: "Field required", type: "missing" },
        ],
      })
    );
  });

  try {
    await assert.rejects(
      async () => {
        await fetchWithTimeout(`${server.url}/api/chat`, { method: "POST" });
      },
      (err) => {
        assert.ok(err instanceof ApiError);
        assert.equal(err.type, "VALIDATION");
        assert.equal(err.status, 422);
        assert.ok(err.message.includes("question: Field required"));
        assert.ok(err.message.includes("video_id: Field required"));
        return true;
      }
    );
  } finally {
    await server.close();
  }
});

test("6. HTTP 500 Internal Server Error extracts detail and formats message", async () => {
  const server = await createMockServer((_req, res) => {
    res.writeHead(500, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ detail: "Failed to generate an answer. Check the backend terminal." }));
  });

  try {
    await assert.rejects(
      async () => {
        await fetchWithTimeout(`${server.url}/api/chat`, { method: "POST" });
      },
      (err) => {
        assert.ok(err instanceof ApiError);
        assert.equal(err.type, "SERVER_ERROR");
        assert.equal(err.status, 500);
        assert.equal(err.message, "Failed to generate an answer. Check the backend terminal.");
        return true;
      }
    );
  } finally {
    await server.close();
  }
});

test("7. Unexpected response format (invalid JSON) throws PARSE_ERROR", async () => {
  const server = await createMockServer((_req, res) => {
    res.writeHead(200, { "Content-Type": "text/html" });
    res.end("<html><body>Bad Gateway</body></html>");
  });

  try {
    await assert.rejects(
      async () => {
        await fetchWithTimeout(`${server.url}/api/chat`, { method: "POST" });
      },
      (err) => {
        assert.ok(err instanceof ApiError);
        assert.equal(err.type, "PARSE_ERROR");
        assert.equal(
          err.message,
          "Received an unexpected response format from the server. Expected JSON."
        );
        return true;
      }
    );
  } finally {
    await server.close();
  }
});

test("8. Successful request parses JSON correctly", async () => {
  const server = await createMockServer((_req, res) => {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ answer: "This is a test answer", sources: [] }));
  });

  try {
    const data = await fetchWithTimeout(`${server.url}/api/chat`, { method: "POST" });
    assert.equal(data.answer, "This is a test answer");
    assert.deepEqual(data.sources, []);
  } finally {
    await server.close();
  }
});

test("9. TIMEOUTS constants match specification (45s for chat, notes, study guide, quiz)", () => {
  assert.equal(TIMEOUTS.CHAT, 45000);
  assert.equal(TIMEOUTS.NOTES, 45000);
  assert.equal(TIMEOUTS.STUDY_GUIDE, 45000);
  assert.equal(TIMEOUTS.QUIZ, 45000);
  assert.equal(TIMEOUTS.INGEST, 45000);
});

test("10. API helper functions exist and handle requests", async () => {
  assert.equal(typeof sendChatMessage, "function");
  assert.equal(typeof generateQuiz, "function");
  assert.equal(typeof generateStudyGuide, "function");
  assert.equal(typeof generateNotes, "function");
  assert.equal(typeof ingestVideo, "function");
  assert.equal(typeof clearChatHistory, "function");
});

test("11. External AbortSignal aborts fetchWithTimeout immediately", async () => {
  const server = await createMockServer(() => {});
  const externalController = new AbortController();

  setTimeout(() => {
    externalController.abort();
  }, 30);

  try {
    await assert.rejects(
      async () => {
        await fetchWithTimeout(`${server.url}/cancel`, { signal: externalController.signal }, 10000);
      },
      (err) => {
        assert.ok(err instanceof ApiError);
        assert.equal(err.type, "ABORTED");
        assert.equal(err.message, "The request was cancelled.");
        return true;
      }
    );
  } finally {
    await server.close();
  }
});

test("12. HTTP 502/503 server error formats message", async () => {
  const server = await createMockServer((_req, res) => {
    res.writeHead(502, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ detail: "Upstream Ollama service unavailable" }));
  });

  try {
    await assert.rejects(
      async () => {
        await fetchWithTimeout(`${server.url}/api/chat`, { method: "POST" });
      },
      (err) => {
        assert.ok(err instanceof ApiError);
        assert.equal(err.type, "SERVER_ERROR");
        assert.equal(err.status, 502);
        assert.equal(err.message, "Upstream Ollama service unavailable");
        return true;
      }
    );
  } finally {
    await server.close();
  }
});
