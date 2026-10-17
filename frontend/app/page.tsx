"use client";

import { useState, useRef, useCallback, useEffect } from "react";

// ─── Types ───────────────────────────────────────────────────────────────

interface EvidenceFrame {
  timestamp_sec: number;
  image_base64: string;
}

interface Violation {
  type: string;
  timecode_sec: number;
  description: string;
  evidence_frames?: EvidenceFrame[];
}

interface JobResult {
  job_id: string;
  status: "queued" | "processing" | "done" | "failed";
  verdict?: "violation" | "no_violation";
  violations?: Violation[];
  error?: string;
}

// ─── Helpers ─────────────────────────────────────────────────────────────

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
const DEFAULT_KEY =
  process.env.NEXT_PUBLIC_DEFAULT_API_KEY || "";

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.round((sec % 1) * 10);
  return `${m}:${s.toString().padStart(2, "0")}.${ms}`;
}

/** Highlight normative references in description text */
function highlightNormRef(text: string): React.ReactNode[] {
  const parts = text.split(
    /(Постановлени[еяю]\s+(?:Правления\s+)?НБРК\s*№\s*\d+[^.]*(?:\.\s*)?)/gi
  );
  return parts.map((part, i) =>
    /Постановлени/i.test(part) ? (
      <span key={i} className="norm-ref">
        📜 {part.trim()}
      </span>
    ) : (
      <span key={i}>{part}</span>
    )
  );
}

// ═════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═════════════════════════════════════════════════════════════════════════

/** Header */
function Header({
  apiKey,
  setApiKey,
}: {
  apiKey: string;
  setApiKey: (k: string) => void;
}) {
  const [showKey, setShowKey] = useState(false);
  return (
    <header className="glass-card-dark text-white">
      <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-forte-primary to-forte-accent flex items-center justify-center text-xl font-bold shadow-lg">
            🛡️
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">CashGuard</h1>
            <p className="text-xs text-graphite-400">
              AI Аудит кассовой дисциплины
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-xs text-graphite-400 mr-1">API Key:</label>
          <div className="relative">
            <input
              id="api-key-input"
              type={showKey ? "text" : "password"}
              value={apiKey}
              onChange={(e) => {
                setApiKey(e.target.value);
                localStorage.setItem("cashguard_api_key", e.target.value);
              }}
              className="bg-graphite-700 border border-graphite-600 rounded-lg px-3 py-1.5 text-sm w-64 focus:outline-none focus:border-forte-primary transition-colors text-white placeholder-graphite-400"
              placeholder="Введите X-API-Key"
            />
            <button
              onClick={() => setShowKey(!showKey)}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-graphite-400 hover:text-white transition-colors text-xs"
            >
              {showKey ? "🙈" : "👁️"}
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}

/** Drag-and-Drop Upload Zone */
function UploadZone({
  onFileSelect,
  file,
  uploading,
}: {
  onFileSelect: (f: File) => void;
  file: File | null;
  uploading: boolean;
}) {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === "dragenter" || e.type === "dragover");
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);
      const f = e.dataTransfer.files[0];
      if (f && (f.name.toLowerCase().endsWith(".mov") || f.name.toLowerCase().endsWith(".mp4"))) {
        onFileSelect(f);
      }
    },
    [onFileSelect]
  );

  return (
    <div
      className={`dropzone rounded-2xl p-8 text-center cursor-pointer transition-all ${
        dragActive ? "dropzone-active" : ""
      }`}
      onDragEnter={handleDrag}
      onDragOver={handleDrag}
      onDragLeave={handleDrag}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        id="file-input"
        type="file"
        accept=".mov,.mp4,video/quicktime,video/mp4"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFileSelect(f);
        }}
      />

      {!file ? (
        <div className="py-6">
          <div className="text-5xl mb-4">📹</div>
          <p className="text-graphite-600 font-medium text-lg mb-1">
            Перетащите видеофайл сюда
          </p>
          <p className="text-graphite-400 text-sm">
            или нажмите для выбора • форматы <code className="bg-graphite-100 px-1.5 py-0.5 rounded text-forte-primary font-mono text-xs">.mov / .mp4</code> • до 100 МБ
          </p>
        </div>
      ) : (
        <div className="flex items-center justify-center gap-3 py-2">
          <div className="text-3xl">🎬</div>
          <div className="text-left">
            <p className="font-semibold text-graphite-900">{file.name}</p>
            <p className="text-sm text-graphite-400">
              {(file.size / 1024 / 1024).toFixed(1)} МБ
            </p>
          </div>
          {uploading && (
            <div className="ml-4">
              <div className="w-5 h-5 border-2 border-forte-primary border-t-transparent rounded-full animate-spin-slow" />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Video Preview Player */
function VideoPlayer({
  file,
  videoRef,
}: {
  file: File;
  videoRef: React.RefObject<HTMLVideoElement | null>;
}) {
  const [url, setUrl] = useState<string>("");

  useEffect(() => {
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);

  if (!url) return null;

  return (
    <div className="glass-card rounded-2xl overflow-hidden mt-4 animate-fade-in-up">
      <video
        ref={videoRef}
        src={url}
        controls
        className="w-full max-h-[400px] bg-black"
      />
    </div>
  );
}

/** Processing Status */
function ProcessingStatus({
  status,
  elapsed,
}: {
  status: string;
  elapsed: number;
}) {
  return (
    <div className="glass-card rounded-2xl p-6 mt-6 animate-fade-in-up">
      <div className="flex items-center gap-4">
        <div className="relative">
          <div className="w-12 h-12 rounded-full bg-gradient-to-br from-forte-primary to-forte-accent flex items-center justify-center">
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin-slow" />
          </div>
          <div className="absolute inset-0 rounded-full border-2 border-forte-primary opacity-50 animate-ping" />
        </div>
        <div>
          <p className="font-semibold text-graphite-900 text-lg">
            {status === "queued" ? "В очереди..." : "Анализ видео..."}
          </p>
          <p className="text-graphite-400 text-sm">
            {status === "queued"
              ? "Ожидание GPU"
              : "VLM анализирует кадры на наличие нарушений"}
          </p>
        </div>
        <div className="ml-auto text-right">
          <p className="text-2xl font-bold text-forte-primary tabular-nums">
            {Math.floor(elapsed)}с
          </p>
          <p className="text-xs text-graphite-400">прошло</p>
        </div>
      </div>
      <div className="mt-4 h-1.5 bg-graphite-100 rounded-full overflow-hidden">
        <div className="h-full bg-gradient-to-r from-forte-primary to-forte-accent animate-shimmer rounded-full w-full" />
      </div>
    </div>
  );
}

/** Verdict Badge */
function VerdictBadge({ verdict }: { verdict: "violation" | "no_violation" }) {
  const isViolation = verdict === "violation";
  return (
    <div
      className={`rounded-2xl p-6 mt-6 animate-fade-in-up text-center ${
        isViolation
          ? "bg-gradient-to-r from-danger/10 to-danger/5 border border-danger/20"
          : "bg-gradient-to-r from-success/10 to-success/5 border border-success/20"
      }`}
    >
      <div className="text-5xl mb-3">{isViolation ? "⚠️" : "✅"}</div>
      <h2
        className={`text-2xl font-bold mb-1 ${
          isViolation ? "text-danger" : "text-success"
        }`}
      >
        {isViolation
          ? "Обнаружены признаки возможных нарушений"
          : "Нарушений не обнаружено"}
      </h2>
      <p className="text-graphite-400 text-sm">
        {isViolation
          ? "Результаты анализа требуют внимания аудитора"
          : "Видеозапись соответствует нормативным требованиям НБРК"}
      </p>
    </div>
  );
}

/** Evidence Frame Modal */
function FrameModal({
  frame,
  onClose,
}: {
  frame: EvidenceFrame;
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="relative max-w-4xl w-full animate-fade-in-up"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute -top-3 -right-3 w-10 h-10 rounded-full bg-white shadow-lg flex items-center justify-center text-graphite-600 hover:text-danger transition-colors text-xl z-10"
        >
          ✕
        </button>
        <img
          src={`data:image/jpeg;base64,${frame.image_base64}`}
          alt={`Evidence at ${frame.timestamp_sec}s`}
          className="w-full rounded-2xl shadow-2xl"
        />
        <div className="absolute bottom-4 left-4 bg-black/60 backdrop-blur-sm text-white px-3 py-1.5 rounded-lg text-sm font-medium">
          ⏱️ {formatTime(frame.timestamp_sec)}
        </div>
      </div>
    </div>
  );
}

/** Single Violation Card */
function ViolationCard({
  violation,
  index,
  onSeek,
}: {
  violation: Violation;
  index: number;
  onSeek: (t: number) => void;
}) {
  const [modalFrame, setModalFrame] = useState<EvidenceFrame | null>(null);

  return (
    <>
      {modalFrame && (
        <FrameModal frame={modalFrame} onClose={() => setModalFrame(null)} />
      )}

      <div
        className="glass-card rounded-2xl p-5 animate-fade-in-up"
        style={{ animationDelay: `${index * 100}ms` }}
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-3 mb-3">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-lg bg-danger/10 flex items-center justify-center text-danger font-bold text-sm shrink-0 mt-0.5">
              {index + 1}
            </div>
            <div>
              <h3 className="font-bold text-graphite-900 text-base leading-snug">
                {violation.type}
              </h3>
            </div>
          </div>
          <button
            id={`seek-btn-${index}`}
            onClick={() => onSeek(violation.timecode_sec)}
            className="shrink-0 flex items-center gap-1.5 bg-forte-primary/10 hover:bg-forte-primary/20 text-forte-primary px-3 py-1.5 rounded-lg text-sm font-semibold transition-colors cursor-pointer"
          >
            ▶ {formatTime(violation.timecode_sec)}
          </button>
        </div>

        {/* Description */}
        <div className="text-sm text-graphite-600 leading-relaxed mb-4 pl-11">
          {highlightNormRef(violation.description)}
        </div>

        {/* Evidence Frames */}
        {violation.evidence_frames && violation.evidence_frames.length > 0 && (
          <div className="pl-11">
            <p className="text-xs text-graphite-400 font-medium mb-2 uppercase tracking-wider">
              Кадры-доказательства
            </p>
            <div className="flex gap-2 overflow-x-auto pb-1">
              {violation.evidence_frames.map((frame, fi) => (
                <button
                  key={fi}
                  onClick={() => setModalFrame(frame)}
                  className="relative shrink-0 group cursor-pointer"
                >
                  <img
                    src={`data:image/jpeg;base64,${frame.image_base64}`}
                    alt={`Evidence ${fi + 1}`}
                    className="w-36 h-24 object-cover rounded-lg border border-graphite-100 group-hover:border-forte-primary transition-colors"
                  />
                  <div className="absolute bottom-1 right-1 bg-black/60 text-white text-[10px] px-1.5 py-0.5 rounded font-mono">
                    {formatTime(frame.timestamp_sec)}
                  </div>
                  <div className="absolute inset-0 bg-forte-primary/0 group-hover:bg-forte-primary/10 rounded-lg transition-colors flex items-center justify-center">
                    <span className="opacity-0 group-hover:opacity-100 text-white text-lg transition-opacity drop-shadow-lg">
                      🔍
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  );
}

// ═════════════════════════════════════════════════════════════════════════
// MAIN PAGE
// ═════════════════════════════════════════════════════════════════════════

export default function HomePage() {
  const [apiKey, setApiKey] = useState(DEFAULT_KEY);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<JobResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);

  // Load API key from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("cashguard_api_key");
    if (saved) setApiKey(saved);
  }, []);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const resetState = () => {
    setJobId(null);
    setResult(null);
    setError(null);
    setElapsed(0);
    if (pollingRef.current) clearInterval(pollingRef.current);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  const handleFileSelect = (f: File) => {
    resetState();
    setFile(f);
  };

  const seekTo = (t: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = t;
      videoRef.current.play();
    }
  };

  // ── Upload & Start Analysis ──────────────────────────────────────────
  const handleAnalyze = async () => {
    if (!file || !apiKey) return;
    resetState();
    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const resp = await fetch(`${API_URL}/api/v1/analyze`, {
        method: "POST",
        headers: { "X-API-Key": apiKey },
        body: formData,
      });

      if (resp.status === 401 || resp.status === 403) {
        setError("Неверный API-ключ. Проверьте X-API-Key.");
        setUploading(false);
        return;
      }

      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        setError(body.detail || `Ошибка ${resp.status}`);
        setUploading(false);
        return;
      }

      const data = await resp.json();
      setJobId(data.job_id);
      setUploading(false);

      // Start timer
      startTimeRef.current = Date.now();
      timerRef.current = setInterval(() => {
        setElapsed((Date.now() - startTimeRef.current) / 1000);
      }, 100);

      // Start polling
      pollingRef.current = setInterval(async () => {
        try {
          const pollResp = await fetch(
            `${API_URL}/api/v1/jobs/${data.job_id}`,
            { headers: { "X-API-Key": apiKey } }
          );
          if (!pollResp.ok) return;
          const pollData: JobResult = await pollResp.json();

          if (pollData.status === "done" || pollData.status === "failed") {
            setResult(pollData);
            if (pollingRef.current) clearInterval(pollingRef.current);
            if (timerRef.current) clearInterval(timerRef.current);
          }
        } catch {
          // Silent retry on network errors
        }
      }, 2000);
    } catch (err) {
      setError("Ошибка сети. Проверьте подключение к серверу.");
      setUploading(false);
    }
  };

  const isProcessing =
    jobId && (!result || result.status === "processing" || result.status === "queued");

  return (
    <div className="min-h-screen flex flex-col">
      <Header apiKey={apiKey} setApiKey={setApiKey} />

      <main className="flex-1 max-w-4xl mx-auto w-full px-4 py-8">
        {/* Upload Section */}
        <section className="glass-card rounded-2xl p-6 mb-6">
          <h2 className="text-xl font-bold text-graphite-900 mb-4 flex items-center gap-2">
            <span className="text-2xl">📤</span> Загрузить видео для анализа
          </h2>

          <UploadZone
            onFileSelect={handleFileSelect}
            file={file}
            uploading={uploading}
          />

          {file && (
            <div className="mt-4 flex justify-end">
              <button
                id="analyze-btn"
                className="btn-forte flex items-center gap-2"
                disabled={uploading || !!isProcessing || !apiKey}
                onClick={handleAnalyze}
              >
                {uploading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin-slow" />
                    Загрузка...
                  </>
                ) : (
                  <>🔍 Начать анализ</>
                )}
              </button>
            </div>
          )}
        </section>

        {/* Video Preview */}
        {file && <VideoPlayer file={file} videoRef={videoRef} />}

        {/* Error */}
        {error && (
          <div className="mt-6 bg-danger/10 border border-danger/20 text-danger rounded-2xl p-4 animate-fade-in-up flex items-center gap-3">
            <span className="text-2xl">❌</span>
            <p className="font-medium">{error}</p>
          </div>
        )}

        {/* Processing Status */}
        {isProcessing && (
          <ProcessingStatus
            status={result?.status || "queued"}
            elapsed={elapsed}
          />
        )}

        {/* Failed */}
        {result?.status === "failed" && (
          <div className="mt-6 bg-danger/10 border border-danger/20 text-danger rounded-2xl p-6 animate-fade-in-up text-center">
            <div className="text-4xl mb-2">💥</div>
            <h3 className="text-lg font-bold mb-1">Ошибка обработки</h3>
            <p className="text-sm">{result.error || "Неизвестная ошибка"}</p>
          </div>
        )}

        {/* Results */}
        {result?.status === "done" && (
          <div className="mt-6 space-y-4">
            <VerdictBadge verdict={result.verdict!} />

            {result.violations && result.violations.length > 0 && (
              <div className="space-y-4 mt-6">
                <h2 className="text-lg font-bold text-graphite-900 flex items-center gap-2">
                  <span className="text-xl">📋</span>
                  Обнаруженные нарушения ({result.violations.length})
                </h2>
                {result.violations.map((v, i) => (
                  <ViolationCard
                    key={i}
                    violation={v}
                    index={i}
                    onSeek={seekTo}
                  />
                ))}
              </div>
            )}

            {/* Meta info */}
            <div className="glass-card rounded-2xl p-4 mt-4 flex items-center justify-between text-sm text-graphite-400">
              <span>Job ID: <code className="font-mono text-graphite-600">{result.job_id}</code></span>
              <span>Время обработки: <strong className="text-graphite-600">{Math.round(elapsed)}с</strong></span>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="text-center py-6 text-sm text-graphite-400">
        <p>
          CashGuard © 2026 • ForteBank Datathon 26 •{" "}
          <span className="text-forte-primary font-medium">
            Qwen2.5-VL-7B-Instruct
          </span>
        </p>
      </footer>
    </div>
  );
}
