"use client";

import { useRef, useState } from "react";

const BACKEND =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8080";

// Fixed defaults — no pickers in the UI.
const TEMPLATE = "white_studio";
const MODE = "ai";

export default function Home() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [resultUrl, setResultUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [drag, setDrag] = useState(false);
  const inputRef = useRef(null);

  function onPick(f) {
    if (!f || !f.type.startsWith("image/")) {
      setError("Please choose an image file.");
      return;
    }
    setError(null);
    setFile(f);
    setResultUrl(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(f));
  }

  async function generate() {
    if (!file) {
      setError("Upload a car image first.");
      return;
    }
    setLoading(true);
    setError(null);
    setResultUrl(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("template", TEMPLATE);
      form.append("mode", MODE);
      const res = await fetch(`${BACKEND}/process`, { method: "POST", body: form });
      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || `Request failed (${res.status})`);
      }
      const blob = await res.blob();
      setResultUrl(URL.createObjectURL(blob));
    } catch (e) {
      setError(e.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="wrap">
      <div className="grid">
        {/* ---------------- Upload ---------------- */}
        <div className="panel">
          <h2>Upload car image</h2>
          <div
            className={`dropzone ${drag ? "drag" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              onPick(e.dataTransfer.files?.[0]);
            }}
          >
            {previewUrl ? (
              <img className="preview-img" src={previewUrl} alt="car preview" />
            ) : (
              <>
                <div style={{ fontSize: 15 }}>Click or drag a car photo here</div>
                <div className="hint">JPG / PNG</div>
              </>
            )}
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => onPick(e.target.files?.[0])}
          />

          <button className="btn" onClick={generate} disabled={loading}>
            {loading ? "Generating…" : "Generate"}
          </button>
          {error && <div className="error">{error}</div>}
        </div>

        {/* ---------------- Result ---------------- */}
        <div className="panel">
          <h2>Result</h2>
          <div className="result-box">
            {loading ? (
              <div className="spinner" />
            ) : resultUrl ? (
              <img src={resultUrl} alt="result" />
            ) : (
              <span>Your image will appear here</span>
            )}
          </div>
          {resultUrl && (
            <a className="download" href={resultUrl} download="autobg-result.png">
              ↓ Download PNG
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
