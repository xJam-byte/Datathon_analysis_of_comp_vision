"use client";

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";

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

const flagNum = (i: number) => String(i + 1).padStart(2, "0");

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

function LockIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden className="shrink-0 mt-[3px]">
      <rect x="3" y="11" width="18" height="11" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  );
}

// ─── Шапка: коллаб-локап ─────────────────────────────────────────────────

function Header({ apiKey, setApiKey, health }: { apiKey: string; setApiKey: (k: string) => void; health: Health }) {
  const [open, setOpen] = useState(false);
  const ready = health?.status === "ok";
  const statusText = health === null ? "Сервер недоступен" : ready ? "Модель готова" : "Модель загружается или недоступна";

  return (
    <header>
      <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] border-b-2 border-ink">
        <div className="bg-forte text-white px-4 sm:px-8 py-3 font-display font-semibold text-[22px] sm:text-[27px] leading-tight">
          ForteBank
        </div>
        <div className="bg-ink text-wheat-light grid place-items-center px-3 sm:px-4 font-display text-2xl" aria-hidden>
          ×
        </div>
        <div className="tex-wheat-fine text-ink px-4 sm:px-8 py-3 font-display italic font-semibold text-[22px] sm:text-[27px] leading-tight">
          Фермеры
        </div>
      </div>

      <div className="bg-surface border-b border-line">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-2.5 flex items-center gap-x-5 gap-y-2 flex-wrap">
          <div className="flex items-baseline gap-3 mr-auto">
            <span className="text-base font-semibold tracking-wide">CashGuard</span>
            <span className="text-sm text-muted hidden sm:inline">проверка кассовых видеозаписей</span>
          </div>

          <span className="flex items-center gap-2 text-sm text-muted" role="status">
            <span className={`w-2 h-2 ${ready ? "bg-clear" : "bg-alert"}`} aria-hidden />
            {statusText}
          </span>

          <div className="relative">
            <button className="btn-ghost text-sm" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
              Ключ API {apiKey ? "задан" : "не задан"}
            </button>
            {open && (
              <div className="absolute right-0 mt-2 w-72 max-w-[calc(100vw-2rem)] panel p-3 shadow-lg z-20">
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
                    } catch { }
                  }}
                  className="w-full border border-ink bg-white px-3 py-2 text-sm"
                  placeholder="Ключ, выданный командой"
                  autoComplete="off"
                />
              </div>
            )}
          </div>
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
      className={`dropzone p-6 sm:p-9 cursor-pointer flex flex-col gap-5 ${active ? "dropzone-active" : ""}`}
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
      <p className="kicker text-wheat-ink">Загрузка записи</p>
      <p className="font-display font-semibold text-[28px] sm:text-[36px] leading-[1.05]">Перетащите видеозапись кассы сюда</p>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
        <span className="btn-primary">Выбрать файл</span>
        <span className="text-muted text-sm">или нажмите, чтобы выбрать файл · .mov или .mp4</span>
      </div>
      <p className="flex gap-2.5 items-start text-sm text-muted border-t border-line pt-4">
        <LockIcon />
        Видео обрабатывается локально на сервере команды, без внешних AI-сервисов. Лица не распознаются.
      </p>
    </div>
  );
}

// ─── Таймлайн-поле доказательств ─────────────────────────────────────────

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
    <div className="mt-4">
      <div className="relative pt-8">
        <div
          className="field"
          role="slider"
          tabIndex={0}
          aria-label="Таймлайн видео с отметками нарушений"
          aria-valuemin={0}
          aria-valuemax={Math.round(duration)}
          aria-valuenow={Math.round(current)}
          aria-valuetext={formatTime(current)}
          onClick={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            onSeek(((e.clientX - r.left) / r.width) * duration);
          }}
          onKeyDown={(e) => {
            const step = e.shiftKey ? 10 : 2;
            if (e.key === "ArrowRight") onSeek(Math.min(duration, current + step));
            else if (e.key === "ArrowLeft") onSeek(Math.max(0, current - step));
            else if (e.key === "Home") onSeek(0);
            else if (e.key === "End") onSeek(duration);
            else return;
            e.preventDefault();
          }}
        >
          <div className="field-played" style={{ width: pct(current) }} />
          {violations.map((v, i) =>
            v.time_range_sec ? (
              <div
                key={`r${i}`}
                className="field-range"
                style={{ left: pct(v.time_range_sec[0]), width: `calc(${pct(v.time_range_sec[1])} - ${pct(v.time_range_sec[0])})` }}
              />
            ) : null
          )}
          <div className="field-playhead" style={{ left: pct(current) }} />
        </div>

        {violations.map((v, i) => (
          <Fragment key={`m${i}`}>
            <span className="flag-pole" style={{ left: pct(v.timecode_sec) }} aria-hidden />
            <button
              className="flag"
              data-active={activeIndex === i}
              data-flip={v.timecode_sec / duration > 0.92}
              style={{ left: pct(v.timecode_sec) }}
              title={`${formatTime(v.timecode_sec)} — ${v.type}`}
              aria-label={`Нарушение ${i + 1} на ${formatTime(v.timecode_sec)}`}
              onClick={(e) => {
                e.stopPropagation();
                onSelect(i);
              }}
            >
              <span className="flag-cloth tnum">{flagNum(i)}</span>
            </button>
          </Fragment>
        ))}
      </div>
      <div className="flex justify-between gap-3 text-xs text-muted tnum mt-1.5">
        <span>0:00.0</span>
        <span className="hidden sm:inline">Клик по полю или ← → для перемотки</span>
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
    <div className="panel p-5 sm:p-6" role="status" aria-live="polite">
      <p className="kicker text-wheat-ink">Идёт анализ</p>
      <div className="flex items-baseline justify-between gap-4 mt-2">
        <p className="font-display font-semibold text-[28px] sm:text-[34px] leading-[1.05]">{title}</p>
        <p className="font-display font-semibold text-[32px] sm:text-[40px] leading-none text-wheat-ink tnum shrink-0">
          {ratio === null ? `${Math.floor(elapsed)} с` : `${Math.round(ratio * 100)}%`}
        </p>
      </div>
      <p className="text-sm text-muted mt-1 tnum">
        {detail}
        {detail && ratio !== null ? " · " : ""}
        {ratio !== null ? `${Math.floor(elapsed)} с` : ""}
      </p>
      <div className="progress-track mt-5">
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

// ─── Вердикт: полоса на всю ширину ───────────────────────────────────────

function Verdict({ job, onExport }: { job: JobResponse; onExport: () => void }) {
  const isViolation = job.verdict === "violation";
  const count = job.violations?.length ?? 0;
  const m = job.meta;

  return (
    <section className={`on-dark text-white border-b-2 border-ink ${isViolation ? "bg-forte" : "bg-clear"}`} aria-live="polite">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-7 sm:py-10 flex flex-col lg:flex-row lg:items-end justify-between gap-6">
        <div className="max-w-3xl">
          <p className="kicker text-white/85">Вердикт</p>
          <h2 className="font-display font-semibold text-[34px] sm:text-[52px] leading-[1] mt-3 text-balance">
            {isViolation ? "Обнаружены признаки возможных нарушений" : "Нарушение не обнаружено"}
          </h2>
          <p className="text-[15px] mt-4">
            {isViolation
              ? `Найдено эпизодов: ${count}. Проверьте каждый по кадрам и видео.`
              : "Модель не нашла на видео признаков нарушений из перечня правил."}
          </p>
          <p className="text-sm text-white/85 mt-2 max-w-[70ch]">
            Результат — подсказка для проверяющего, а не юридическое или дисциплинарное заключение. Сервис оценивает только
            то, что видно на видео.
          </p>
          {!!m?.windows_failed && (
            <p className="text-sm text-wheat-light font-medium mt-2">
              {m.windows_failed} фрагм. не удалось проверить — часть видео могла остаться без анализа.
            </p>
          )}
        </div>
        <div className="flex flex-col items-start lg:items-end gap-3 shrink-0">
          <button className="btn-on-dark" onClick={onExport}>
            Скачать отчёт JSON
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
              <path d="M12 3v12" />
              <path d="m7 10 5 5 5-5" />
              <path d="M5 21h14" />
            </svg>
          </button>
          <p className="text-xs text-white/85 tnum lg:text-right">
            {m?.processing_time_sec !== undefined && `Анализ: ${Math.round(m.processing_time_sec)} с`}
            {m?.windows_analyzed !== undefined && ` · проверено фрагментов: ${m.windows_analyzed} из ${m.windows_total}`}
            {` · задача ${job.job_id}`}
          </p>
        </div>
      </div>
    </section>
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
    <article id={`finding-${index}`} className="panel finding p-5 sm:p-6" data-active={active}>
      <div className="flex items-center justify-between gap-3">
        <p className="kicker text-forte">Флажок {flagNum(index)}</p>
        <button
          className="shrink-0 tnum text-sm font-semibold text-forte border border-forte hover:bg-forte hover:text-white px-3 py-1.5 transition-colors"
          onClick={() => onSeek(v.timecode_sec)}
          title="Перемотать видео к этому моменту"
        >
          ▶ {formatTime(v.timecode_sec)}
        </button>
      </div>
      <h3 className="font-display font-semibold text-[24px] sm:text-[28px] leading-[1.1] mt-2">{v.type}</h3>

      <div className="mt-4 space-y-4">
        {observation && (
          <div>
            <p className="text-xs text-muted mb-0.5">Что видно на видео</p>
            <p className="text-[15px]">{observation}</p>
          </div>
        )}

        {source && (
          <div className="bg-vault px-4 py-3">
            <p className="font-display italic font-semibold text-[20px] leading-tight text-forte">
              Правило{v.rule_code ? ` ${v.rule_code}` : ""}
            </p>
            <p className="text-sm mt-1">{source}</p>
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
                    className="w-full aspect-video object-cover border-2 border-transparent group-hover:border-forte"
                  />
                  <span className="absolute bottom-1 left-1 bg-ink/80 text-white text-[11px] px-1.5 tnum">
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
    <div className="on-dark fixed inset-0 z-50 bg-ink/85 flex items-center justify-center p-4" onClick={onClose} role="dialog" aria-modal>
      <figure className="max-w-5xl w-full" onClick={(e) => e.stopPropagation()}>
        <img src={`data:image/jpeg;base64,${frame.image_base64}`} alt="Кадр-доказательство" className="w-full" />
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
      // eslint-disable-next-line react-hooks/set-state-in-effect -- localStorage доступен только на клиенте
      if (saved) setApiKey(saved);
    } catch { }
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

  const videoUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);
  useEffect(() => () => { if (videoUrl) URL.revokeObjectURL(videoUrl); }, [videoUrl]);

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

      <main className="flex-1 w-full">
        {!file ? (
          <section className="grid lg:grid-cols-2 border-b-2 border-ink">
            <div className="on-dark bg-forte-deep text-white px-5 sm:px-10 lg:px-14 py-10 sm:py-16 flex flex-col justify-between gap-10 lg:min-h-[560px]">
              <p className="kicker text-wheat-light">Лимитированный коллаб · 01</p>
              <div>
                <h1 className="font-display font-semibold text-[clamp(52px,8.5vw,116px)] leading-[0.92] tracking-[-0.01em]">
                  Каждая купюра <em className="font-medium text-wheat-light">на&nbsp;виду</em>
                </h1>
                <p className="mt-6 max-w-[48ch] text-[16px] text-white/90">
                  Загрузите запись с камеры кассового узла. Сервис найдёт эпизоды, похожие на нарушения кассовой дисциплины,
                  покажет момент на видео, кадры и пункт нормативного акта НБРК.
                </p>
              </div>
            </div>
            <div className="tex-wheat px-4 sm:px-10 lg:px-14 py-10 sm:py-16 flex flex-col justify-center gap-4">
              <UploadZone onFile={chooseFile} onReject={setError} />
              {error && (
                <p role="alert" className="bg-surface text-alert text-sm font-medium px-4 py-3">
                  {error}
                </p>
              )}
            </div>
          </section>
        ) : (
          <>
            {done && job && <Verdict job={job} onExport={exportJson} />}

            <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 sm:py-8 grid lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)] gap-6 items-start">
              {/* Левая колонка: видео и таймлайн-поле */}
              <section className="lg:sticky lg:top-4">
                <video
                  ref={videoRef}
                  src={videoUrl}
                  controls
                  playsInline
                  className="w-full bg-ink aspect-video"
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
                <div className="flex items-center justify-between gap-3 mt-5 pt-4 border-t-2 border-ink flex-wrap">
                  <div className="min-w-0">
                    <p className="font-medium truncate">{file.name}</p>
                    <p className="text-xs text-muted tnum">
                      {(file.size / 1024 / 1024).toFixed(1)} МБ{duration ? ` · ${formatTime(duration)}` : ""}
                    </p>
                  </div>
                  <div className="flex gap-2 flex-wrap">
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
              </section>

              {/* Правая колонка: статус и находки */}
              <section className="space-y-4" aria-live="polite">
                {error && (
                  <div className="panel p-5 border-alert">
                    <p className="font-semibold text-alert">Анализ не выполнен</p>
                    <p className="text-sm mt-1">{error}</p>
                  </div>
                )}

                {!job && !error && !busy && (
                  <div className="panel p-5 sm:p-6">
                    <p className="kicker text-wheat-ink">Готово к проверке</p>
                    <p className="font-display font-semibold text-[28px] leading-[1.05] mt-2">Видео готово к проверке</p>
                    <p className="text-sm text-muted mt-2">
                      Нажмите «Начать анализ». Результат появится здесь, а найденные моменты — отметками на таймлайне под
                      видео.
                    </p>
                  </div>
                )}

                {busy && <ProcessingPanel job={job} elapsed={elapsed} />}

                {job?.status === "failed" && (
                  <div className="panel p-5 border-alert">
                    <p className="font-semibold text-alert">Не удалось обработать видео</p>
                    <p className="text-sm mt-1 break-words">{job.error || "Неизвестная ошибка сервера."}</p>
                    <p className="text-sm text-muted mt-2">Проверьте статус модели в шапке и запустите анализ снова.</p>
                  </div>
                )}

                {done && job && !violations.length && (
                  <div className="panel p-5 sm:p-6">
                    <p className="font-display italic font-semibold text-[26px] leading-tight">Поле чистое: флажков нет.</p>
                  </div>
                )}

                {done &&
                  job &&
                  violations.map((v, i) => (
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
              </section>
            </div>
          </>
        )}
      </main>

      <footer className="border-t-2 border-ink text-xs text-muted py-5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex flex-wrap justify-between gap-2">
          <span>
            <span className="text-forte font-semibold">ForteBank</span> × <span className="italic font-semibold text-ink">Фермеры</span>
          </span>
          <span>CashGuard · ForteBank Datathon 26 · Qwen2.5-VL-32B-Instruct (AWQ), работает локально</span>
        </div>
      </footer>
    </div>
  );
}
