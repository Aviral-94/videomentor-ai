import { useEffect, useId, useState } from "react";
import mermaid from "mermaid";
import ReactMarkdown from "react-markdown";
import {
  Document,
  Packer,
  Paragraph,
  TextRun,
  HeadingLevel,
  AlignmentType,
} from "docx";

import "./StudyGuide.css";

mermaid.initialize({
  startOnLoad: false,
  securityLevel: "strict",
  theme: "default",
  flowchart: {
    useMaxWidth: true,
    htmlLabels: false,
    curve: "basis",
  },
});

// Convert timestamps such as [1:43] into direct YouTube links.
function makeTimestampsClickable(notes, videoId) {
  if (!notes || !videoId) return notes || "";

  return notes.replace(
    /\[(\d{1,2}):([0-5]\d)(?::([0-5]\d))?\](?!\()/g,
    (match, first, second, third) => {
      let seconds;

      if (third !== undefined) {
        seconds =
          Number(first) * 3600 +
          Number(second) * 60 +
          Number(third);
      } else {
        seconds = Number(first) * 60 + Number(second);
      }

      const youtubeUrl =
        `https://www.youtube.com/watch?v=${encodeURIComponent(videoId)}&t=${seconds}s&autoplay=1`;

      return `[⏱️ ${match.slice(1, -1)}](${youtubeUrl})`;
    }
  );
}

// Render the Mermaid flowchart.
function MermaidDiagram({ chart }) {
  const generatedId = useId();
  const [svg, setSvg] = useState("");
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function renderDiagram() {
      if (!chart?.trim()) {
        setSvg("");
        return;
      }

      try {
        const id = `mermaid-${generatedId.replace(/:/g, "")}`;
        const result = await mermaid.render(id, chart.trim());

        if (!cancelled) {
          setSvg(result.svg);
          setError(false);
        }
      } catch (err) {
        console.error("Mermaid rendering failed:", err);

        if (!cancelled) {
          setError(true);
          setSvg("");
        }
      }
    }

    renderDiagram();

    return () => {
      cancelled = true;
    };
  }, [chart, generatedId]);

  if (error) {
    return (
      <div className="study-guide-diagram-error">
        <p>Unable to render this diagram.</p>

        <details>
          <summary>View diagram source</summary>
          <pre>{chart}</pre>
        </details>
      </div>
    );
  }

  if (!svg) {
    return (
      <div className="study-guide-diagram-loading">
        Rendering diagram...
      </div>
    );
  }

  return (
    <div
      className="study-guide-diagram"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}

export default function StudyGuide({
  videoId,
  onJumpToTimestamp,
  onClose,
}) {
  const [studyGuide, setStudyGuide] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [exportingDocx, setExportingDocx] = useState(false);

  // Generate the study guide.
  async function generateStudyGuide() {
    if (!videoId || loading) return;

    setLoading(true);
    setError("");
    setStudyGuide(null);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/api/study-guide",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            video_id: videoId,
          }),
        }
      );

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          data.detail || "Could not generate the study guide."
        );
      }

      if (!data.notes || !data.flowchart) {
        throw new Error(
          "The server returned incomplete study-guide data."
        );
      }

      setStudyGuide(data);
    } catch (err) {
      console.error("Study guide request failed:", err);
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  // Download Markdown.
  function downloadMarkdown() {
    if (!studyGuide) return;

    const content = [
      studyGuide.notes,
      "",
      "## Visual Flowchart",
      "",
      "```mermaid",
      studyGuide.flowchart,
      "```",
    ].join("\n");

    const blob = new Blob([content], {
      type: "text/markdown;charset=utf-8",
    });

    const url = URL.createObjectURL(blob);
    const anchor = window.document.createElement("a");

    anchor.href = url;
    anchor.download = `videomentor-study-guide-${videoId}.md`;

    window.document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();

    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // Escape HTML before writing notes into the print document.
  function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, (character) => {
      const entities = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      };

      return entities[character];
    });
  }

  // Export study notes and the flowchart to PDF.
  function downloadPDF() {
    if (!studyGuide) return;

    try {
      const notesHTML = window.document.querySelector(
        ".study-guide-markdown"
      )?.innerHTML || "";

      const diagramElement = window.document.querySelector(
        ".study-guide-diagram svg"
      );

      const diagramHTML = diagramElement
        ? diagramElement.outerHTML
        : `<pre>${escapeHTML(studyGuide.flowchart)}</pre>`;

      const printWindow = window.open("", "_blank");

      if (!printWindow) {
        alert(
          "The print window was blocked. Allow pop-ups for this site and try again."
        );
        return;
      }

      printWindow.document.open();

      printWindow.document.write(`
        <!DOCTYPE html>
        <html>
          <head>
            <meta charset="UTF-8" />
            <title>VideoMentor AI Study Guide</title>

            <style>
              @page {
                size: A4;
                margin: 18mm;
              }

              body {
                font-family: Arial, sans-serif;
                color: #222;
                line-height: 1.65;
              }

              h1, h2, h3 {
                color: #5145cd;
                page-break-after: avoid;
              }

              h1 {
                border-bottom: 2px solid #6c63ff;
                padding-bottom: 10px;
              }

              p, li, pre {
                overflow-wrap: anywhere;
              }

              pre, code {
                white-space: pre-wrap;
              }

              a {
                color: #5145cd;
                text-decoration: none;
              }

              .diagram {
                margin-top: 24px;
                page-break-inside: avoid;
              }

              .diagram svg {
                display: block;
                max-width: 100%;
                height: auto;
                margin: 0 auto;
              }

              .footer {
                margin-top: 35px;
                font-size: 11px;
                color: #777;
              }
            </style>
          </head>

          <body>
            <h1>VideoMentor AI Study Guide</h1>

            <p>
              <strong>YouTube Video ID:</strong>
              ${escapeHTML(videoId)}
            </p>

            <hr />

            <h2>Study Notes</h2>
            ${notesHTML}

            <div class="diagram">
              <h2>Visual Flowchart</h2>
              ${diagramHTML}
            </div>

            <div class="footer">
              Generated by VideoMentor AI
            </div>
          </body>
        </html>
      `);

      printWindow.document.close();

      printWindow.onload = () => {
        printWindow.focus();
        printWindow.print();
      };
    } catch (err) {
      console.error("PDF export failed:", err);
      setError("Could not prepare the PDF. Please try again.");
    }
  }

  // Convert Markdown notes into editable Word paragraphs.
  function convertNotesToWord(notes) {
    const lines = String(notes || "").split("\n");
    const paragraphs = [];
    let insideCodeBlock = false;

    for (const originalLine of lines) {
      const line = originalLine.trim();

      if (line.startsWith("```")) {
        insideCodeBlock = !insideCodeBlock;
        continue;
      }

      if (!line) {
        paragraphs.push(
          new Paragraph({
            children: [new TextRun("")],
            spacing: { after: 100 },
          })
        );
        continue;
      }

      if (insideCodeBlock) {
        paragraphs.push(
          new Paragraph({
            children: [
              new TextRun({
                text: line,
                font: "Consolas",
                size: 20,
              }),
            ],
            spacing: { after: 80 },
          })
        );
        continue;
      }

      const headingMatch = line.match(/^(#{1,3})\s+(.*)$/);

      if (headingMatch) {
        const level = headingMatch[1].length;

        paragraphs.push(
          new Paragraph({
            text: headingMatch[2],
            heading:
              level === 1
                ? HeadingLevel.HEADING_1
                : level === 2
                  ? HeadingLevel.HEADING_2
                  : HeadingLevel.HEADING_3,
            spacing: { before: 180, after: 100 },
          })
        );

        continue;
      }

      if (/^[-*]\s+/.test(line)) {
        paragraphs.push(
          new Paragraph({
            text: line.replace(/^[-*]\s+/, ""),
            bullet: { indent: 360 },
            spacing: { after: 80 },
          })
        );

        continue;
      }

      if (/^\d+\.\s+/.test(line)) {
        paragraphs.push(
          new Paragraph({
            text: line.replace(/^\d+\.\s+/, ""),
            indent: { left: 360 },
            spacing: { after: 80 },
          })
        );

        continue;
      }

      const cleanLine = line
        .replace(
          /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
          "$1 ($2)"
        )
        .replace(/\*\*(.*?)\*\*/g, "$1")
        .replace(/__(.*?)__/g, "$1")
        .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "$1")
        .replace(/`([^`]+)`/g, "$1");

      paragraphs.push(
        new Paragraph({
          children: [new TextRun(cleanLine)],
          spacing: { after: 100 },
        })
      );
    }

    return paragraphs;
  }

  // Download the editable Word document.
  async function downloadDOCX() {
    if (!studyGuide || exportingDocx) return;

    setExportingDocx(true);
    setError("");

    let objectUrl;

    try {
      const children = [
        new Paragraph({
          text: "VideoMentor AI Study Guide",
          heading: HeadingLevel.TITLE,
          alignment: AlignmentType.CENTER,
          spacing: { after: 300 },
        }),

        new Paragraph({
          children: [
            new TextRun({
              text: `YouTube Video ID: ${videoId}`,
              italics: true,
              color: "666666",
            }),
          ],
          spacing: { after: 300 },
        }),

        new Paragraph({
          text: "Study Notes",
          heading: HeadingLevel.HEADING_1,
          spacing: { before: 200, after: 150 },
        }),

        ...convertNotesToWord(studyGuide.notes),

        new Paragraph({
          text: "Visual Flowchart",
          heading: HeadingLevel.HEADING_1,
          pageBreakBefore: true,
          spacing: { before: 300, after: 150 },
        }),

        new Paragraph({
          children: [
            new TextRun({
              text:
                "Mermaid flowchart source:",
              italics: true,
              color: "666666",
            }),
          ],
          spacing: { after: 150 },
        }),

        ...String(studyGuide.flowchart || "")
          .split("\n")
          .map(
            (line) =>
              new Paragraph({
                children: [
                  new TextRun({
                    text: line || " ",
                    font: "Consolas",
                    size: 20,
                  }),
                ],
                spacing: { after: 60 },
              })
          ),

        new Paragraph({
          children: [
            new TextRun({
              text: "Generated by VideoMentor AI",
              color: "888888",
              size: 18,
            }),
          ],
          alignment: AlignmentType.CENTER,
          spacing: { before: 400 },
        }),
      ];

      // IMPORTANT: Use "doc", not "document".
      // This prevents shadowing the browser's document object.
      const doc = new Document({
        creator: "VideoMentor AI",
        title: "VideoMentor AI Study Guide",
        description: "AI-generated YouTube learning notes",
        sections: [
          {
            properties: {},
            children,
          },
        ],
      });

      const blob = await Packer.toBlob(doc);

      objectUrl = URL.createObjectURL(blob);

      const anchor = window.document.createElement("a");

      anchor.href = objectUrl;
      anchor.download = `videomentor-study-guide-${videoId}.docx`;

      window.document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();

      setTimeout(() => {
        if (objectUrl) {
          URL.revokeObjectURL(objectUrl);
        }
      }, 1500);
    } catch (err) {
      console.error("DOCX export failed:", err);

      setError(
        `Could not create the Word document: ${
          err?.message || "Unknown error"
        }`
      );
    } finally {
      setExportingDocx(false);
    }
  }

  // Render Markdown links.
  const markdownComponents = {
    a({ href, children, ...props }) {
      const isYouTubeTimestamp =
        href?.includes("youtube.com/watch") ||
        href?.includes("youtu.be/");

      return (
        <a
          href={href}
          {...props}
          target="_blank"
          rel="noopener noreferrer"
          className={
            isYouTubeTimestamp
              ? "study-guide-timestamp"
              : undefined
          }
        >
          {children}
        </a>
      );
    },
  };

  const clickableNotes = makeTimestampsClickable(
    studyGuide?.notes || "",
    videoId
  );

  return (
    <section className="study-guide-card">
      <div className="study-guide-header">
        <div>
          <span className="study-guide-eyebrow">
            AI LEARNING TOOLS
          </span>

          <h2>Visual Study Guide</h2>

          <p>
            Turn your video into structured notes and visual diagrams.
          </p>
        </div>

        {onClose && (
          <button
            type="button"
            className="study-guide-close"
            onClick={onClose}
            aria-label="Close study guide"
          >
            ✕
          </button>
        )}
      </div>

      <div className="study-guide-actions">
        <button
          type="button"
          className="study-guide-generate"
          onClick={generateStudyGuide}
          disabled={!videoId || loading}
        >
          {loading
            ? "Generating study guide..."
            : studyGuide
              ? "Regenerate Study Guide"
              : "✨ Generate Study Guide"}
        </button>

        {studyGuide && (
          <>
            <button
              type="button"
              className="study-guide-download"
              onClick={downloadMarkdown}
            >
              ↓ Download Markdown
            </button>

            <button
              type="button"
              className="study-guide-download"
              onClick={downloadPDF}
            >
              📄 Download PDF
            </button>

            <button
              type="button"
              className="study-guide-download"
              onClick={downloadDOCX}
              disabled={exportingDocx}
            >
              {exportingDocx
                ? "Creating Word document..."
                : "📝 Download DOCX"}
            </button>
          </>
        )}
      </div>

      {loading && (
        <div className="study-guide-status">
          <div className="study-guide-spinner" />

          <p>
            Analyzing the transcript and generating your notes and
            diagrams. This may take a little while.
          </p>
        </div>
      )}

      {error && (
        <div className="study-guide-error" role="alert">
          <strong>Export or generation failed</strong>

          <p>{error}</p>

          <button
            type="button"
            onClick={generateStudyGuide}
            disabled={loading}
          >
            Try again
          </button>
        </div>
      )}

      {studyGuide && !loading && (
        <div className="study-guide-content">
          <div className="study-guide-section">
            <h3>📝 Your Study Notes</h3>

            <article className="study-guide-markdown">
              <ReactMarkdown components={markdownComponents}>
                {clickableNotes}
              </ReactMarkdown>
            </article>
          </div>

          <div className="study-guide-section">
            <h3>🔀 Visual Flowchart</h3>

            <p className="study-guide-description">
              Explore the process or relationships between the important
              concepts in your video.
            </p>

            <MermaidDiagram chart={studyGuide.flowchart} />
          </div>

          <div className="study-guide-footer">
            <span>
              Transcript segments retrieved:{" "}
              {studyGuide.transcript_segments_used ?? "N/A"}
            </span>

            <span>Powered by local AI</span>
          </div>
        </div>
      )}
    </section>
  );
}