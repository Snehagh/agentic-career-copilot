import { useState } from "react";
import { uploadFile, analyze } from "./api";
import "./App.css";

const STEPS = { UPLOAD: "upload", ANALYZING: "analyzing", RESULTS: "results" };

function parseScore(raw) {
  try {
    const cleaned = raw.replace(/```json|```/g, "").trim();
    return JSON.parse(cleaned);
  } catch {
    return null;
  }
}

export default function App() {
  const [resume, setResume] = useState(null);
  const [job, setJob] = useState(null);
  const [jobTitle, setJobTitle] = useState("");
  const [step, setStep] = useState(STEPS.UPLOAD);
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");
  const [uploadStatus, setUploadStatus] = useState({ resume: false, job: false });

  async function handleUpload(file, type) {
    setError("");
    try {
      await uploadFile(file, type);
      setUploadStatus((s) => ({ ...s, [type]: true }));
    } catch (e) {
      setError(e.message);
    }
  }

  async function handleAnalyze() {
    if (!uploadStatus.resume || !uploadStatus.job) {
      setError("Please upload both a resume and a job description first.");
      return;
    }
    if (!jobTitle.trim()) {
      setError("Please enter the job title.");
      return;
    }
    setError("");
    setStep(STEPS.ANALYZING);
    try {
      const data = await analyze(jobTitle);
      setResults(data);
      setStep(STEPS.RESULTS);
    } catch (e) {
      setError(e.message);
      setStep(STEPS.UPLOAD);
    }
  }

  function reset() {
    setResume(null);
    setJob(null);
    setJobTitle("");
    setResults(null);
    setError("");
    setUploadStatus({ resume: false, job: false });
    setStep(STEPS.UPLOAD);
  }

  const scoreData = results ? parseScore(results.match_score) : null;
  const score = scoreData?.score ?? null;

  return (
    <div className="shell">
      <header>
        <div className="logo">⚡ Career Copilot</div>
        <p className="tagline">AI-powered resume-to-job match analysis</p>
      </header>

      <main>
        {step === STEPS.UPLOAD && (
          <div className="card upload-card">
            <h2>Get your match score</h2>
            <p className="sub">Upload your resume and a job description to see how well you fit the role.</p>

            <div className="upload-row">
              <UploadBox
                label="Resume"
                icon="📄"
                accept=".pdf,.txt"
                done={uploadStatus.resume}
                onChange={(f) => { setResume(f); handleUpload(f, "resume"); }}
              />
              <UploadBox
                label="Job Description"
                icon="💼"
                accept=".pdf,.txt"
                done={uploadStatus.job}
                onChange={(f) => { setJob(f); handleUpload(f, "job"); }}
              />
            </div>

            <div className="field">
              <label htmlFor="title">Job title</label>
              <input
                id="title"
                type="text"
                placeholder="e.g. AI Engineer"
                value={jobTitle}
                onChange={(e) => setJobTitle(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAnalyze()}
              />
            </div>

            {error && <p className="error">{error}</p>}

            <button
              className="btn-primary"
              onClick={handleAnalyze}
              disabled={!uploadStatus.resume || !uploadStatus.job || !jobTitle.trim()}
            >
              Analyze Match →
            </button>
          </div>
        )}

        {step === STEPS.ANALYZING && (
          <div className="card center">
            <div className="spinner" />
            <p>Running multi-agent analysis…</p>
            <p className="sub">Resume Analyst → JD Analyst → Match Scorer → Career Coach</p>
          </div>
        )}

        {step === STEPS.RESULTS && results && (
          <div className="results">
            <ScoreRing score={score} />

            <div className="grid-2">
              <Section title="✅ Matched Skills" content={scoreData?.matched_skills} list />
              <Section title="⚠️ Skill Gaps" content={scoreData?.gaps} list />
            </div>

            <Section title="📋 Resume Analysis" content={results.resume_analysis} />
            <Section title="🔍 Job Requirements" content={results.job_analysis} />
            <Section title="🎯 Coaching Recommendations" content={results.recommendations} />

            <button className="btn-secondary" onClick={reset}>← Start over</button>
          </div>
        )}
      </main>

      <footer>
        <span>FastAPI · ChromaDB · sentence-transformers · React</span>
      </footer>
    </div>
  );
}

function UploadBox({ label, icon, accept, done, onChange }) {
  return (
    <label className={`upload-box ${done ? "done" : ""}`}>
      <span className="upload-icon">{done ? "✓" : icon}</span>
      <span className="upload-label">{done ? `${label} uploaded` : `Upload ${label}`}</span>
      <span className="upload-hint">{accept.replace(/\./g, "").toUpperCase()}</span>
      <input
        type="file"
        accept={accept}
        hidden
        onChange={(e) => e.target.files[0] && onChange(e.target.files[0])}
      />
    </label>
  );
}

function ScoreRing({ score }) {
  const pct = score ?? 0;
  const r = 54;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ;
  const color = pct >= 80 ? "#22c55e" : pct >= 60 ? "#f59e0b" : "#ef4444";

  return (
    <div className="score-ring-wrap">
      <svg width="140" height="140" viewBox="0 0 140 140">
        <circle cx="70" cy="70" r={r} fill="none" stroke="#e5e7eb" strokeWidth="12" />
        <circle
          cx="70" cy="70" r={r} fill="none"
          stroke={color} strokeWidth="12"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          transform="rotate(-90 70 70)"
        />
      </svg>
      <div className="score-label">
        <span className="score-number" style={{ color }}>{score ?? "—"}</span>
        <span className="score-sub">/ 100 match</span>
      </div>
    </div>
  );
}

function Section({ title, content, list }) {
  if (!content || (Array.isArray(content) && content.length === 0)) return null;
  return (
    <div className="section-card">
      <h3>{title}</h3>
      {list && Array.isArray(content) ? (
        <ul>{content.map((item, i) => <li key={i}>{item}</li>)}</ul>
      ) : (
        <pre>{content}</pre>
      )}
    </div>
  );
}
