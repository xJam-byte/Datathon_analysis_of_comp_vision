"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// ─── Типы (схема API, раздел 10 ТЗ + доп. поля бэкенда) ─────────────────

interface EvidenceFrame {
  timestamp_sec: number;
  image_base64: string;
}

interface Violation {
  type: string;
  timecode_sec: number;
  description: string;
  evidence_frames?: EvidenceFrame[];
  rule_code?: string;
  normative_source?: string;
  observation?: string;
  time_range_sec?: [number, number];
}

interface JobMeta {
  duration_sec?: number;
  windows_total?: number;
  windows_analyzed?: number;
  windows_failed?: number;
  processing_time_sec?: number;
}

interface JobResponse {
  job_id: string;
  status: "processing" | "done" | "failed";
  stage?: "queued" | "processing";
  queue_position?: number;
  progress?: { done: number; total: number };
  verdict?: "violation" | "no_violation";
  violations?: Violation[];
  meta?: JobMeta;
  error?: string;
}

type Health = { status: string; model?: string } | null;

/**
 * Адрес бэкенда определяется в браузере, а не при сборке:
 *  - NEXT_PUBLIC_API_URL, если задан при сборке;
 *  - RunPod-прокси: https://<pod>-3000.proxy.runpod.net → https://<pod>-8080.proxy.runpod.net;
 *  - иначе тот же хост, порт 8080.
 */
function resolveApiUrl(): string {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "");
  if (typeof window === "undefined") return "http://localhost:8080";
  const { protocol, hostname } = window.location;
  if (/-3000\./.test(hostname)) return `${protocol}//${hostname.replace("-3000.", "-8080.")}`;
  return `${protocol}//${hostname}:8080`;
}
const DEFAULT_KEY = process.env.NEXT_PUBLIC_DEFAULT_API_KEY || "";
const ACCEPTED = [".mov", ".mp4"];
const POLL_MS = 2000;
const POLL_GIVE_UP_MS = 8 * 60 * 1000;

// ─── Утилиты ─────────────────────────────────────────────────────────────

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

function isAccepted(name: string) {
  const n = name.toLowerCase();
  return ACCEPTED.some((ext) => n.endsWith(ext));
}

/** Отделяет наблюдение от ссылки на норму для старых ответов без доп. полей */
function splitDescription(v: Violation): { observation: string; source: string } {
  if (v.observation !== undefined && v.normative_source) {
    return { observation: v.observation, source: v.normative_source };
  }
  const idx = v.description.search(/Нормативн\w*\s+источник\s*:/i);
  if (idx === -1) return { observation: v.description, source: v.normative_source || "" };
  return {
    observation: v.description.slice(0, idx).trim(),
    source: v.description.slice(idx).replace(/^Нормативн\w*\s+источник\s*:\s*/i, "").replace(/\.$/, ""),
  };
}

// ─── Шапка ───────────────────────────────────────────────────────────────

function Header({ apiKey, setApiKey, health }: { apiKey: string; setApiKey: (k: string) => void; health: Health }) {
  const [open, setOpen] = useState(false);
  const ready = health?.status === "ok";
  const statusText = health === null ? "Сервер недоступен" : ready ? "Модель готова" : "Модель загружается или недоступна";

  return (
    <header className="bg-surface border-b border-line">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center gap-4 flex-wrap">
        <div className="flex items-baseline gap-3 mr-auto">
          <span className="text-lg font-semibold tracking-tight">CashGuard</span>
          <span className="text-sm text-muted hidden sm:inline">проверка кассовых видеозаписей</span>
        </div>

        <span className="flex items-center gap-2 text-sm text-muted" role="status">
          <span className={`w-2 h-2 rounded-full ${ready ? "bg-clear" : "bg-alert"}`} aria-hidden />
          {statusText}
        </span>

        <div className="relative">
          <button className="btn-ghost text-sm" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
            Ключ API {apiKey ? "задан" : "не задан"}
          </button>
          {open && (
            <div className="absolute right-0 mt-2 w-72 panel p-3 shadow-lg z-20">
              <label htmlFor="api-key" className="block text-sm text-muted mb-1">
                X-API-Key
              </label>
              <input
                id="api-key"
                type="password"
                value={apiKey}
                onChange={(e) => {
                  setApiKey(e.target.value);
                  try {
                    localStorage.setItem("cashguard_api_key", e.target.value);
                  } catch {}
                }}
                className="w-full border border-line rounded-md px-3 py-2 text-sm"
                placeholder="Ключ, выданный командой"
                autoComplete="off"
              />
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

// ─── Загрузка ────────────────────────────────────────────────────────────

function UploadZone({ onFile, onReject }: { onFile: (f: File) => void; onReject: (msg: string) => void }) {
  const [active, setActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const take = (f: File | undefined) => {
    if (!f) return;
    if (!isAccepted(f.name)) {
      onReject(`Файл «${f.name}» не подходит: нужен .mov или .mp4.`);
      return;
    }
    onFile(f);
  };

  return (
    <div
      role="button"
      tabIndex={0}
      className={`dropzone p-10 text-center cursor-pointer ${active ? "dropzone-active" : ""}`}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setActive(true);
      }}
      onDragLeave={() => setActive(false)}
      onDrop={(e) => {
        e.preventDefault();
        setActive(false);
        take(e.dataTransfer.files[0]);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".mov,.mp4,video/quicktime,video/mp4"
        className="hidden"
        onChange={(e) => {
          take(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      <p className="text-lg font-medium">Перетащите видеозапись кассы сюда</p>
      <p className="text-muted text-sm mt-1">или нажмите, чтобы выбрать файл · .mov или .mp4</p>
    </div>
  );
}

// ─── Таймлайн доказательств ──────────────────────────────────────────────

function EvidenceTimeline({
  duration,
  current,
  violations,
  activeIndex,
  onSelect,
  onSeek,
}: {
  duration: number;
  current: number;
  violations: Violation[];
  activeIndex: number | null;
  onSelect: (i: number) => void;
  onSeek: (t: number) => void;
}) {
  if (!duration) return null;
  const pct = (t: number) => `${Math.min(100, Math.max(0, (t / duration) * 100))}%`;

  return (
    <div className="mt-3">
      <div
        className="timeline"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          onSeek(((e.clientX - r.left) / r.width) * duration);
        }}
        aria-label="Таймлайн видео с отметками нарушений"
      >
        {violations.map((v, i) =>
          v.time_range_sec ? (
            <div
              key={`r${i}`}
              className="timeline-range"
              style={{ left: pct(v.time_range_sec[0]), width: `calc(${pct(v.time_range_sec[1])} - ${pct(v.time_range_sec[0])})` }}
            />
          ) : null
        )}
        {violations.map((v, i) => (
          <button
            key={`m${i}`}
            className="timeline-mark"
            data-active={activeIndex === i}
            style={{ left: pct(v.timecode_sec) }}
            title={`${formatTime(v.timecode_sec)} — ${v.type}`}
            aria-label={`Нарушение ${i + 1} на ${formatTime(v.timecode_sec)}`}
            onClick={(e) => {
              e.stopPropagation();
              onSelect(i);
            }}
          />
        ))}
        <div className="timeline-playhead" style={{ left: pct(current) }} />
      </div>
      <div className="flex justify-between text-xs text-muted tnum mt-1">
        <span>0:00.0</span>
        <span>{formatTime(duration)}</span>
      </div>
    </div>
  );
}

// ─── Статус обработки ────────────────────────────────────────────────────

function ProcessingPanel({ job, elapsed }: { job: JobResponse | null; elapsed: number }) {
  const p = job?.progress;
  const queued = job?.stage === "queued";
  const ratio = p && p.total > 0 ? p.done / p.total : null;

  let title = "Загружаем видео на сервер";
  let detail = "";
  if (job) {
    if (queued) {
      title = "Видео в очереди";
      detail = job.queue_position ? `Перед ним ещё ${job.queue_position - 1} видео` : "Скоро начнём";
    } else if (!p || p.total === 0) {
      title = "Извлекаем кадры из видео";
    } else {
      title = "Модель просматривает видео";
      detail = `Проверено фрагментов: ${p.done} из ${p.total}`;
    }
  }

  return (
    <div className="panel p-5" role="status" aria-live="polite">
      <div className="flex items-baseline justify-between gap-4">
        <p className="font-semibold">{title}</p>
        <p className="text-muted text-sm tnum">{Math.floor(elapsed)} с</p>
      </div>
      {detail && <p className="text-sm text-muted mt-1">{detail}</p>}
      <div className="progress-track mt-4">
        {ratio === null ? (
          <div className="progress-bar progress-indeterminate" />
        ) : (
          <div className="progress-bar" style={{ width: `${Math.max(4, ratio * 100)}%` }} />
        )}
      </div>
      <p className="text-xs text-muted mt-3">Обычно анализ занимает 1–3 минуты. Страницу можно не обновлять.</p>
    </div>
  );
}

// ─── Вердикт ─────────────────────────────────────────────────────────────

function Verdict({ job, onExport }: { job: JobResponse; onExport: () => void }) {
  const isViolation = job.verdict === "violation";
  const count = job.violations?.length ?? 0;
  const m = job.meta;

  return (
    <div className={`rounded-[10px] p-5 border ${isViolation ? "bg-forte-soft border-forte/30" : "bg-clear-soft border-clear/30"}`}>
      <p className={`text-xl font-semibold ${isViolation ? "text-forte" : "text-clear"}`}>
        {isViolation ? "Обнаружены признаки возможных нарушений" : "Нарушение не обнаружено"}
      </p>
      <p className="text-sm mt-1">
        {isViolation
          ? `Найдено эпизодов: ${count}. Проверьте каждый по кадрам и видео.`
          : "Модель не нашла на видео признаков нарушений из перечня правил."}
      </p>
      <p className="text-xs text-muted mt-3">
        Результат — подсказка для проверяющего, а не юридическое или дисциплинарное заключение. Сервис оценивает только
        то, что видно на видео.
      </p>
      <div className="flex items-center justify-between gap-3 mt-4 flex-wrap">
        <p className="text-xs text-muted tnum">
          {m?.processing_time_sec !== undefined && `Анализ: ${Math.round(m.processing_time_sec)} с`}
          {m?.windows_analyzed !== undefined && ` · проверено фрагментов: ${m.windows_analyzed} из ${m.windows_total}`}
          {` · задача ${job.job_id}`}
        </p>
        <button className="btn-ghost text-sm" onClick={onExport}>
          Скачать отчёт JSON
        </button>
      </div>
      {!!m?.windows_failed && (
        <p className="text-xs text-alert mt-2">
          {m.windows_failed} фрагм. не удалось проверить — часть видео могла остаться без анализа.
        </p>
      )}
    </div>
  );
}

// ─── Карточка нарушения ──────────────────────────────────────────────────

function FindingCard({
  v,
  index,
  active,
  onSeek,
  onOpenFrame,
}: {
  v: Violation;
  index: number;
  active: boolean;
  onSeek: (t: number) => void;
  onOpenFrame: (f: EvidenceFrame) => void;
}) {
  const { observation, source } = splitDescription(v);

  return (
    <article id={`finding-${index}`} className="panel finding p-5" data-active={active}>
      <div className="flex items-start gap-3">
        <span className="tnum text-sm text-muted pt-0.5 w-5 shrink-0">{index + 1}</span>
        <h3 className="font-semibold leading-snug flex-1">{v.type}</h3>
        <button
          className="shrink-0 tnum text-sm font-semibold text-forte bg-forte-soft hover:bg-forte hover:text-white px-2.5 py-1 rounded-md transition-colors"
          onClick={() => onSeek(v.timecode_sec)}
          title="Перемотать видео к этому моменту"
        >
          ▶ {formatTime(v.timecode_sec)}
        </button>
      </div>

      <div className="pl-8 mt-3 space-y-3">
        {observation && (
          <div>
            <p className="text-xs text-muted mb-0.5">Что видно на видео</p>
            <p className="text-[15px]">{observation}</p>
          </div>
        )}

        {source && (
          <div className="border-l-2 border-forte pl-3">
            <p className="text-xs text-muted mb-0.5">Правило{v.rule_code ? ` ${v.rule_code}` : ""}</p>
            <p className="text-sm">{source}</p>
          </div>
        )}

        {!!v.evidence_frames?.length && (
          <div>
            <p className="text-xs text-muted mb-1.5">Кадры-доказательства</p>
            <div className="grid grid-cols-3 gap-2">
              {v.evidence_frames.map((f, i) => (
                <button key={i} className="relative group" onClick={() => onOpenFrame(f)} title="Открыть кадр">
                  <img
                    src={`data:image/jpeg;base64,${f.image_base64}`}
                    alt={`Кадр на ${formatTime(f.timestamp_sec)}`}
                    className="w-full aspect-video object-cover rounded-md border border-line group-hover:border-forte"
                  />
                  <span className="absolute bottom-1 left-1 bg-ink/75 text-white text-[11px] px-1.5 rounded tnum">
                    {formatTime(f.timestamp_sec)}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </article>
  );
}

function FrameModal({ frame, onClose }: { frame: EvidenceFrame; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 bg-ink/80 flex items-center justify-center p-4" onClick={onClose} role="dialog" aria-modal>
      <figure className="max-w-5xl w-full" onClick={(e) => e.stopPropagation()}>
        <img src={`data:image/jpeg;base64,${frame.image_base64}`} alt="Кадр-доказательство" className="w-full rounded-md" />
        <figcaption className="flex justify-between items-center text-white mt-2 text-sm">
          <span className="tnum">Кадр на {formatTime(frame.timestamp_sec)}</span>
          <button className="underline" onClick={onClose}>
            Закрыть
          </button>
        </figcaption>
      </figure>
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════

export default function HomePage() {
  const [apiKey, setApiKey] = useState(DEFAULT_KEY);
  const [health, setHealth] = useState<Health>({ status: "unknown" });
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [duration, setDuration] = useState(0);
  const [current, setCurrent] = useState(0);
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState<JobResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [modalFrame, setModalFrame] = useState<EvidenceFrame | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const timers = useRef<{ poll?: ReturnType<typeof setInterval>; tick?: ReturnType<typeof setInterval> }>({});

  const stopTimers = () => {
    clearInterval(timers.current.poll);
    clearInterval(timers.current.tick);
  };

  useEffect(() => {
    try {
      const saved = localStorage.getItem("cashguard_api_key");
      if (saved) setApiKey(saved);
    } catch {}
    return stopTimers;
  }, []);

  // Состояние сервиса
  useEffect(() => {
    const load = async () => {
      try {
        const r = await fetch(`${resolveApiUrl()}/health`);
        setHealth(r.ok ? await r.json() : null);
      } catch {
        setHealth(null);
      }
    };
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!file) return;
    const u = URL.createObjectURL(file);
    setVideoUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);

  const reset = () => {
    stopTimers();
    setJob(null);
    setError(null);
    setElapsed(0);
    setActiveIndex(null);
  };

  const chooseFile = (f: File) => {
    reset();
    setDuration(0);
    setFile(f);
  };

  const seek = useCallback((t: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = Math.max(0, t);
    v.pause();
  }, []);

  const selectFinding = useCallback(
    (i: number) => {
      setActiveIndex(i);
      const v = job?.violations?.[i];
      if (v) seek(v.timecode_sec);
      document.getElementById(`finding-${i}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    },
    [job, seek]
  );

  const analyze = async () => {
    if (!file) return;
    if (!apiKey) {
      setError("Укажите ключ API в правом верхнем углу.");
      return;
    }
    reset();
    setBusy(true);
    const started = Date.now();
    timers.current.tick = setInterval(() => setElapsed((Date.now() - started) / 1000), 250);

    try {
      const form = new FormData();
      form.append("file", file);
      const resp = await fetch(`${resolveApiUrl()}/api/v1/analyze`, { method: "POST", headers: { "X-API-Key": apiKey }, body: form });
      if (resp.status === 401 || resp.status === 403) throw new Error("Ключ API не подошёл. Проверьте его в правом верхнем углу.");
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body.detail || `Сервер ответил ошибкой ${resp.status}.`);
      }
      const { job_id } = await resp.json();
      setJob({ job_id, status: "processing", stage: "queued" });

      timers.current.poll = setInterval(async () => {
        if (Date.now() - started > POLL_GIVE_UP_MS) {
          stopTimers();
          setBusy(false);
          setError("Анализ идёт дольше 8 минут. Проверьте состояние сервера и запустите анализ заново.");
          return;
        }
        try {
          const r = await fetch(`${resolveApiUrl()}/api/v1/jobs/${job_id}`, { headers: { "X-API-Key": apiKey } });
          if (r.status === 404) {
            stopTimers();
            setBusy(false);
            setError("Сервер не знает эту задачу — скорее всего, он перезапускался. Запустите анализ заново.");
            return;
          }
          if (!r.ok) return;
          const data: JobResponse = await r.json();
          setJob(data);
          if (data.status !== "processing") {
            stopTimers();
            setBusy(false);
            if (data.violations?.length) setActiveIndex(0);
          }
        } catch {
          // сеть моргнула — пробуем на следующем тике
        }
      }, POLL_MS);
    } catch (e) {
      stopTimers();
      setBusy(false);
      setError(e instanceof Error && e.message !== "Failed to fetch" ? e.message : "Нет связи с сервером. Проверьте адрес и подключение.");
    }
  };

  const exportJson = () => {
    if (!job) return;
    const blob = new Blob([JSON.stringify(job, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `cashguard_${job.job_id}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const done = job?.status === "done";
  const violations = done ? job?.violations ?? [] : [];

  return (
    <div className="min-h-screen flex flex-col">
      <Header apiKey={apiKey} setApiKey={setApiKey} health={health} />
      {modalFrame && <FrameModal frame={modalFrame} onClose={() => setModalFrame(null)} />}

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6">
        {!file ? (
          <section className="max-w-3xl">
            <h1 className="text-2xl font-semibold tracking-tight">Проверка видеозаписи кассы</h1>
            <p className="text-muted mt-1 mb-5 max-w-[62ch]">
              Загрузите запись с камеры кассового узла. Сервис найдёт эпизоды, похожие на нарушения кассовой дисциплины,
              покажет момент на видео, кадры и пункт нормативного акта НБРК.
            </p>
            <UploadZone onFile={chooseFile} onReject={setError} />
            {error && <p className="text-alert text-sm mt-3">{error}</p>}
            <p className="text-xs text-muted mt-4 max-w-[62ch]">
              Видео обрабатывается локально на сервере команды, без внешних AI-сервисов. Лица не распознаются.
            </p>
          </section>
        ) : (
          <div className="grid lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)] gap-6 items-start">
            {/* Левая колонка: видео и таймлайн */}
            <section className="lg:sticky lg:top-4">
              <div className="panel p-3">
                <video
                  ref={videoRef}
                  src={videoUrl}
                  controls
                  playsInline
                  className="w-full rounded-md bg-ink aspect-video"
                  onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || 0)}
                  onTimeUpdate={(e) => setCurrent(e.currentTarget.currentTime)}
                />
                <EvidenceTimeline
                  duration={job?.meta?.duration_sec || duration}
                  current={current}
                  violations={violations}
                  activeIndex={activeIndex}
                  onSelect={selectFinding}
                  onSeek={seek}
                />
                <div className="flex items-center justify-between gap-3 mt-3 flex-wrap">
                  <div className="min-w-0">
                    <p className="font-medium truncate">{file.name}</p>
                    <p className="text-xs text-muted tnum">
                      {(file.size / 1024 / 1024).toFixed(1)} МБ{duration ? ` · ${formatTime(duration)}` : ""}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <label className="btn-ghost text-sm cursor-pointer">
                      Другое видео
                      <input
                        type="file"
                        accept=".mov,.mp4,video/quicktime,video/mp4"
                        className="hidden"
                        disabled={busy}
                        onChange={(e) => {
                          const f = e.target.files?.[0];
                          if (f && isAccepted(f.name)) chooseFile(f);
                          else if (f) setError(`Файл «${f.name}» не подходит: нужен .mov или .mp4.`);
                          e.target.value = "";
                        }}
                      />
                    </label>
                    <button className="btn-primary" onClick={analyze} disabled={busy}>
                      {busy ? "Идёт анализ…" : done || job?.status === "failed" ? "Проверить снова" : "Начать анализ"}
                    </button>
                  </div>
                </div>
              </div>
            </section>

            {/* Правая колонка: статус, вердикт, находки */}
            <section className="space-y-4" aria-live="polite">
              {error && (
                <div className="panel p-4 border-alert/40">
                  <p className="font-semibold text-alert">Анализ не выполнен</p>
                  <p className="text-sm mt-1">{error}</p>
                </div>
              )}

              {!job && !error && !busy && (
                <div className="panel p-5">
                  <p className="font-semibold">Видео готово к проверке</p>
                  <p className="text-sm text-muted mt-1">
                    Нажмите «Начать анализ». Результат появится здесь, а найденные моменты — отметками на таймлайне под
                    видео.
                  </p>
                </div>
              )}

              {busy && <ProcessingPanel job={job} elapsed={elapsed} />}

              {job?.status === "failed" && (
                <div className="panel p-5 border-alert/40">
                  <p className="font-semibold text-alert">Не удалось обработать видео</p>
                  <p className="text-sm mt-1 break-words">{job.error || "Неизвестная ошибка сервера."}</p>
                  <p className="text-sm text-muted mt-2">Проверьте статус модели в шапке и запустите анализ снова.</p>
                </div>
              )}

              {done && job && (
                <>
                  <Verdict job={job} onExport={exportJson} />
                  {violations.map((v, i) => (
                    <FindingCard
                      key={i}
                      v={v}
                      index={i}
                      active={activeIndex === i}
                      onSeek={(t) => {
                        setActiveIndex(i);
                        seek(t);
                      }}
                      onOpenFrame={setModalFrame}
                    />
                  ))}
                </>
              )}
            </section>
          </div>
        )}
      </main>

      <footer className="text-xs text-muted text-center py-5">
        CashGuard · ForteBank Datathon 26 · Qwen2.5-VL-7B-Instruct + LoRA, работает локально
      </footer>
    </div>
  );
}
