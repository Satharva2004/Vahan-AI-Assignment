"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";

type ModelSpec = {
  id: string;
  provider: string;
  label: string;
  rationale: string;
};

type Aggregate = {
  model_id: string;
  provider: string;
  label: string;
  recordings: number;
  wer: number | null;
  cer: number | null;
  entity_f1: number | null;
  entity_recall: number | null;
  latency_ms: number | null;
  failures: number;
};

type RunResult = {
  model_id: string;
  label: string;
  transcript: string;
  ok: boolean;
  error?: string;
  metrics?: {
    wer: number;
    cer: number;
    entity_f1: number;
    entity_recall: number;
    missed_entities: string[];
  };
};

type Run = {
  file_name: string;
  reference: string;
  results: RunResult[];
};

type BenchmarkResponse = {
  ok: boolean;
  error?: string;
  aggregate: Aggregate[];
  runs: Run[];
  methodology: Record<string, string>;
  failure_analysis?: FailureAnalysis;
};

type RecordingRow = {
  id: string;
  file?: File;
  previewUrl?: string;
  reference: string;
  entities: string;
  condition: string;
};

type FailureAnalysis = {
  by_condition: Record<string, { total: number; failed_models: number; missed_entities: number }>;
  by_model: Record<string, { total: number; failures: number; missed_entities: number; avg_wer: number | null }>;
  by_entity: Record<string, number>;
};

type ProgressItem = {
  modelId: string;
  label: string;
  status: "queued" | "running" | "done" | "failed";
  elapsedMs?: number;
  error?: string;
};

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const baselineModelId = "deepgram-nova-3";
const conditionOptions = ["Quiet", "Traffic", "Phone call", "Rushed", "Whispered", "Noisy room", "Hinglish", "Kannada"];

const defaultModels: ModelSpec[] = [
  { id: "deepgram-nova-3", provider: "deepgram", label: "Deepgram Nova-3", rationale: "" },
  { id: "sarvam-saaras-v3-transcribe", provider: "sarvam", label: "Sarvam Saaras v3", rationale: "" },
  { id: "sarvam-saaras-v3-codemix", provider: "sarvam", label: "Sarvam Saaras v3 Codemix", rationale: "" },
  { id: "assemblyai-best", provider: "assemblyai", label: "AssemblyAI Best", rationale: "" },
  { id: "google-stt-long", provider: "google", label: "Google STT Long", rationale: "" },
  { id: "openai-gpt-4o-mini-transcribe", provider: "openai", label: "OpenAI GPT-4o Mini Transcribe", rationale: "" },
];

function newRow(index: number): RecordingRow {
  return { id: `recording-${index}`, reference: "", entities: "", condition: "Quiet" };
}

function formatPct(value: number | null | undefined) {
  if (value == null) return "n/a";
  return `${Math.round(value * 1000) / 10}%`;
}

export default function Home() {
  const [rows, setRows] = useState<RecordingRow[]>([newRow(0)]);
  const [models, setModels] = useState<ModelSpec[]>(defaultModels);
  const [selected, setSelected] = useState<string[]>([baselineModelId]);
  const [modelToAdd, setModelToAdd] = useState("sarvam-saaras-v3-codemix");
  const [result, setResult] = useState<BenchmarkResponse | null>(null);
  const [progress, setProgress] = useState<ProgressItem[]>([]);
  const [recordingRowId, setRecordingRowId] = useState<string | null>(null);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    fetch(`${apiBase}/models`)
      .then((response) => response.json())
      .then((payload) => {
        const fetched = payload.models ?? defaultModels;
        setModels(fetched);
        setModelToAdd(fetched.find((model: ModelSpec) => model.id !== baselineModelId && !selected.includes(model.id))?.id ?? "");
      })
      .catch(() => setModels(defaultModels));
  }, [selected]);

  const selectedModels = useMemo(
    () => selected.map((id) => models.find((model) => model.id === id)).filter(Boolean) as ModelSpec[],
    [models, selected],
  );

  function updateRow(id: string, patch: Partial<RecordingRow>) {
    setRows((current) => current.map((row) => (row.id === id ? { ...row, ...patch } : row)));
  }

  function updateFile(id: string, event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    updateRow(id, { file, previewUrl: file ? URL.createObjectURL(file) : undefined });
  }

  async function startRecording(id: string) {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunksRef.current = [];
      const mimeType = MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : MediaRecorder.isTypeSupported("audio/mp4")
          ? "audio/mp4"
          : "";
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorderRef.current = recorder;
      setRecordingRowId(id);
      setRecordingSeconds(0);
      timerRef.current = window.setInterval(() => setRecordingSeconds((seconds) => seconds + 1), 1000);

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        const type = (recorder.mimeType || "audio/webm").split(";", 1)[0];
        const blob = new Blob(chunksRef.current, { type });
        const extension = type.includes("mp4") ? "m4a" : "webm";
        const file = new File([blob], `recording-${Date.now()}.${extension}`, { type });
        updateRow(id, { file, previewUrl: URL.createObjectURL(file) });
        stream.getTracks().forEach((track) => track.stop());
        if (timerRef.current) window.clearInterval(timerRef.current);
        timerRef.current = null;
        recorderRef.current = null;
        setRecordingRowId(null);
      };
      recorder.start();
    } catch {
      setError("Microphone permission is required to record audio.");
      setRecordingRowId(null);
    }
  }

  function stopRecording() {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
  }

  function addRecording() {
    setRows((current) => [...current, newRow(current.length)].slice(0, 20));
  }

  function selectModel(modelId: string) {
    setModelToAdd(modelId);
    if (modelId && modelId !== baselineModelId && !selected.includes(modelId)) {
      setSelected((current) => [...current, modelId]);
    }
  }

  async function runBenchmark(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    setProgress([]);

    try {
      const usable = rows.filter((row) => row.file && row.reference.trim());
      if (!usable.length) throw new Error("Upload audio and add the actual text.");
      const comparisonModels = selected.filter((id) => id !== baselineModelId);
      if (comparisonModels.length < 1) {
        throw new Error("Select at least one model to compare against Deepgram Nova-3.");
      }
      const runOrder = [baselineModelId, ...comparisonModels];
      setProgress(
        runOrder.map((modelId) => ({
          modelId,
          label: models.find((model) => model.id === modelId)?.label ?? modelId,
          status: "queued",
        })),
      );

      let merged: BenchmarkResponse | null = null;
      for (const modelId of runOrder) {
        const label = models.find((model) => model.id === modelId)?.label ?? modelId;
        const started = performance.now();
        setProgress((current) => current.map((item) => item.modelId === modelId ? { ...item, status: "running" } : item));

        const form = new FormData();
        usable.forEach((row) => form.append("files", row.file as File));
        form.append("references", JSON.stringify(usable.map((row) => row.reference)));
        form.append("entities", JSON.stringify(usable.map((row) => row.entities)));
        form.append("conditions", JSON.stringify(usable.map((row) => row.condition)));
        form.append("model_ids", JSON.stringify([modelId]));

        const response = await fetch(`${apiBase}/benchmark`, { method: "POST", body: form });
        const payload = (await response.json()) as BenchmarkResponse;
        const elapsedMs = Math.round(performance.now() - started);

        if (!response.ok || !payload.ok) {
          const message = payload.error ?? `${label} failed.`;
          setProgress((current) => current.map((item) => item.modelId === modelId ? { ...item, status: "failed", elapsedMs, error: message } : item));
          continue;
        }

        merged = mergeBenchmarkResults(merged, payload);
        setResult(merged);
        setProgress((current) => current.map((item) => item.modelId === modelId ? { ...item, status: "done", elapsedMs } : item));
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Benchmark failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-background text-foreground">
      <header className="bg-background">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-10">
          <p className="font-display text-3xl tracking-tight">AI ASR Lab</p>
          <nav className="hidden items-center gap-10 text-base font-bold md:flex">
            <span>Upload</span>
            <span>Compare</span>
            <span>Report</span>
            <button type="button" className="button-secondary h-14 px-5">Assignment</button>
          </nav>
        </div>
      </header>

      <section className="mx-auto max-w-4xl px-6 pb-18 pt-16 text-center">
        <h1 className="font-display text-3xl leading-tight md:text-4xl">Automatic Speech Recognition Benchmarking</h1>
        <p className="mx-auto mt-4 max-w-2xl text-xl leading-6 theme-muted">
          Upload audio, add the actual sentence, and compare ASR models.
        </p>
      </section>

      <form onSubmit={runBenchmark} className="mx-auto max-w-6xl px-6 pb-16">
        <section className="assignment-shell relative overflow-hidden">
          {loading ? <RunAnimation /> : null}

          <div className="mb-8 flex flex-col justify-between gap-5 md:flex-row md:items-start">
            <div className="flex items-center gap-5">
              <span className="card-icon">
                <UploadGlyph />
              </span>
              <div>
                <p className="text-sm font-bold theme-muted">Benchmark input</p>
                <h2 className="font-display text-2xl">Upload and compare</h2>
              </div>
            </div>

            <div className="grid gap-3 md:min-w-[420px]">
              <div className="grid gap-3 sm:grid-cols-[1fr_auto]">
                <select value={modelToAdd} onChange={(event) => selectModel(event.target.value)} className="field">
                  {models.filter((model) => model.id !== baselineModelId).map((model) => (
                    <option key={model.id} value={model.id}>{model.label}</option>
                  ))}
                </select>
                <button className="button-primary px-6" disabled={loading}>
                  {loading ? "Running" : "Run"}
                </button>
              </div>
              <div className="flex flex-wrap justify-end gap-2">
                {selectedModels.map((model) => (
                  <button
                    key={model.id}
                    type="button"
                    onClick={() => {
                      if (model.id !== baselineModelId) {
                        setSelected((current) => current.filter((id) => id !== model.id));
                      }
                    }}
                    className={`model-pill ${model.id === baselineModelId ? "model-pill-fixed" : ""}`}
                  >
                    {model.id === baselineModelId ? `${model.label} baseline` : model.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-6">
            {progress.length ? <ProgressPanel progress={progress} result={result} /> : null}

            {rows.map((row, index) => (
              <div key={row.id} className="recording-card">
                <div className="flex items-start justify-between gap-4 md:hidden">
                  <p className="font-display text-xl">File {index + 1}</p>
                  {index === rows.length - 1 && rows.length < 20 ? (
                    <button type="button" className="button-secondary" onClick={addRecording}>Add file</button>
                  ) : null}
                </div>

                <div className="grid gap-5 lg:grid-cols-[1fr_1.15fr]">
                  <div className="space-y-3">
                    <div className="hidden items-center justify-between md:flex">
                      <p className="font-display text-xl">File {index + 1}</p>
                      {index === rows.length - 1 && rows.length < 20 ? (
                        <button type="button" className="button-secondary" onClick={addRecording}>Add file</button>
                      ) : null}
                    </div>

                    <div className={`upload-zone ${row.file ? "upload-zone-ready" : ""}`}>
                      <label className="upload-hit-area">
                        <input type="file" accept="audio/*,video/mp4,video/webm" onChange={(event) => updateFile(row.id, event)} />
                        <span className="upload-icon">
                          <UploadGlyph />
                        </span>
                        <span className="font-display text-2xl">{row.file ? "File selected" : "Choose or record audio"}</span>
                        <span className="max-w-xs truncate text-sm theme-muted">{row.file?.name ?? "WAV, MP3, M4A, WebM, or MP4"}</span>
                        <span className="browse-chip">Browse</span>
                      </label>

                      <div className="record-controls">
                        {recordingRowId === row.id ? (
                          <button type="button" className="record-button recording" onClick={stopRecording}>
                            Stop {recordingSeconds}s
                          </button>
                        ) : (
                          <button type="button" className="record-button" onClick={() => startRecording(row.id)} disabled={Boolean(recordingRowId)}>
                            Record now
                          </button>
                        )}
                      </div>

                      {row.previewUrl ? <audio className="audio-preview" src={row.previewUrl} controls /> : null}
                    </div>
                  </div>

                  <div className="grid content-start gap-4">
                    <label>
                      <span className="mb-2 block text-sm font-bold">Actual text</span>
                      <textarea
                        value={row.reference}
                        onChange={(event) => updateRow(row.id, { reference: event.target.value })}
                        placeholder="Type the sentence spoken in the audio"
                        className="field min-h-36 resize-y"
                      />
                    </label>

                    <label>
                      <span className="mb-2 block text-sm font-bold">Entities</span>
                      <input
                        value={row.entities}
                        onChange={(event) => updateRow(row.id, { entities: event.target.value })}
                        placeholder="Places or names, comma separated"
                        className="field"
                      />
                    </label>

                    <div>
                      <span className="mb-2 block text-sm font-bold">Condition</span>
                      <div className="condition-grid">
                        {conditionOptions.map((condition) => (
                          <button
                            key={condition}
                            type="button"
                            className={`condition-bubble ${row.condition === condition ? "condition-bubble-active" : ""}`}
                            onClick={() => updateRow(row.id, { condition })}
                          >
                            {condition}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {error ? <p className="text-sm font-bold text-[#b91c1c]">{error}</p> : null}
          </div>
        </section>
      </form>

      {result ? <Report result={result} /> : null}
    </main>
  );
}

function mergeBenchmarkResults(current: BenchmarkResponse | null, next: BenchmarkResponse): BenchmarkResponse {
  if (!current) return next;
  return {
    ...current,
    aggregate: [...current.aggregate, ...next.aggregate].sort((a, b) => (a.wer ?? 999) - (b.wer ?? 999)),
    runs: current.runs.map((run, index) => ({
      ...run,
      results: [...run.results, ...(next.runs[index]?.results ?? [])],
    })),
  };
}

function ProgressPanel({ progress, result }: { progress: ProgressItem[]; result: BenchmarkResponse | null }) {
  return (
    <div className="progress-panel">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-sm font-bold theme-muted">Live model responses</p>
          <h3 className="font-display text-xl">Processing</h3>
        </div>
        <span className="text-sm font-bold theme-muted">{progress.filter((item) => item.status === "done" || item.status === "failed").length}/{progress.length}</span>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-2">
        {progress.map((item) => {
          const modelResult = result?.aggregate.find((aggregate) => aggregate.model_id === item.modelId);
          return (
            <div key={item.modelId} className="progress-row">
              <div>
                <p className="font-bold">{item.label}</p>
                <p className="text-sm theme-muted">
                  {item.status === "queued" ? "Waiting" : null}
                  {item.status === "running" ? "Running..." : null}
                  {item.status === "done" ? `Done in ${item.elapsedMs} ms` : null}
                  {item.status === "failed" ? `Failed in ${item.elapsedMs} ms` : null}
                </p>
              </div>
              <div className="text-right text-sm font-bold">
                {item.status === "done" ? formatPct(modelResult?.wer) : null}
                {item.status === "running" ? <span className="tiny-loader" /> : null}
                {item.status === "failed" ? <span className="text-[#b91c1c]">Error</span> : null}
              </div>
              {item.error ? <p className="col-span-full text-xs text-[#b91c1c]">{item.error}</p> : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function RunAnimation() {
  return (
    <div className="absolute inset-0 z-10 grid place-items-center border-2 border-foreground bg-background/95">
      <div className="text-center">
        <div className="mx-auto mb-5 flex h-20 w-20 items-center justify-center rounded-full border-2 border-foreground bg-[#a78bfa] shadow-[6px_6px_0_0_var(--foreground)]">
          <span className="pulse-dot" />
        </div>
        <p className="font-display text-2xl">Running benchmark</p>
        <p className="mt-2 text-sm theme-muted">Comparing transcripts now</p>
      </div>
    </div>
  );
}

function UploadGlyph() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 16V4" strokeLinecap="round" />
      <path d="m7 9 5-5 5 5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" strokeLinecap="round" />
    </svg>
  );
}

function Report({ result }: { result: BenchmarkResponse }) {
  const best = result.aggregate.find((item) => item.failures === 0) ?? result.aggregate[0];
  const [analysis, setAnalysis] = useState<string>("");
  const [analysisSource, setAnalysisSource] = useState<string>("");
  const benchmarkDimensions = [
    "Transcript accuracy",
    "Character/spelling accuracy",
    "Entity recall",
    "Latency",
    "Failure rate",
    "Transcript evidence",
  ];

  useEffect(() => {
    fetch(`${apiBase}/analysis`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(result),
    })
      .then((response) => response.json())
      .then((payload) => {
        setAnalysis(payload.summary ?? "Analysis unavailable.");
        setAnalysisSource(payload.source ?? "");
      })
      .catch(() => setAnalysis("Analysis unavailable."));
  }, [result]);

  return (
    <section className="mx-auto max-w-5xl space-y-6 px-5 pb-14">
      <h2 className="font-display text-3xl">Results</h2>

      <div className="theme-panel">
        <div className="flex flex-col justify-between gap-3 md:flex-row md:items-center">
          <div>
            <p className="text-sm font-bold theme-muted">Recommendation</p>
            <h3 className="font-display text-xl">{best?.label ?? "No winner yet"}</h3>
          </div>
          <p className="max-w-xl text-sm theme-muted">Choose the model with the lowest entity-heavy error, not only the lowest WER. Locality names are the product risk.</p>
        </div>
      </div>

      <div className="theme-panel">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-bold theme-muted">{analysisSource ? `${analysisSource} analysis` : "Analysis"}</p>
            <h3 className="font-display text-xl">Answerable summary</h3>
          </div>
        </div>
        <p className="mt-4 whitespace-pre-line text-sm leading-6 theme-muted">{analysis || "Generating analysis..."}</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="theme-panel">
          <h3 className="font-display text-xl">Model quality share</h3>
          <PieChart items={result.aggregate} />
        </div>

        <div className="theme-panel">
          <h3 className="font-display text-xl">Latency by model</h3>
          <LatencyChart items={result.aggregate} />
        </div>
      </div>

      <div className="theme-panel">
        <h3 className="font-display text-xl">Benchmarking against</h3>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {benchmarkDimensions.map((dimension) => (
            <div key={dimension} className="benchmark-chip">{dimension}</div>
          ))}
        </div>
      </div>

      <div className="theme-panel">
        <h3 className="font-display text-xl">Benchmark table</h3>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[760px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-y-2 border-foreground theme-muted">
                <th className="py-2 pr-4">Rank</th>
                <th className="py-2 pr-4">Model</th>
                <th className="py-2 pr-4">WER</th>
                <th className="py-2 pr-4">CER</th>
                <th className="py-2 pr-4">Entity recall</th>
                <th className="py-2 pr-4">Latency</th>
                <th className="py-2 pr-4">Status</th>
              </tr>
            </thead>
            <tbody>
              {result.aggregate.map((item, index) => (
                <tr key={item.model_id} className="border-b border-foreground/10 align-top">
                  <td className="py-3 pr-4 font-bold">{index + 1}</td>
                  <td className="py-3 pr-4 font-bold">{item.label}</td>
                  <td className="py-3 pr-4">{formatPct(item.wer)}</td>
                  <td className="py-3 pr-4">{formatPct(item.cer)}</td>
                  <td className="py-3 pr-4">{formatPct(item.entity_recall)}</td>
                  <td className="py-3 pr-4">{item.latency_ms ? `${item.latency_ms} ms` : "n/a"}</td>
                  <td className="py-3 pr-4">{item.failures ? `${item.failures} failed` : "OK"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="theme-panel">
        <h3 className="font-display text-xl">Transcript evidence</h3>
        <div className="mt-4 space-y-5">
          {result.runs.map((run, index) => (
            <details key={`${run.file_name}-${index}`} open={index === 0}>
              <summary className="cursor-pointer font-bold">{run.file_name || `File ${index + 1}`}</summary>
              <p className="mt-2 text-sm theme-muted">Actual: {run.reference}</p>
              <div className="mt-3 overflow-x-auto">
                <table className="w-full min-w-[720px] border-collapse text-left text-sm">
                  <thead>
                    <tr className="border-y-2 border-foreground theme-muted">
                      <th className="py-2 pr-4">Model</th>
                      <th className="py-2 pr-4">WER</th>
                      <th className="py-2 pr-4">Missed</th>
                      <th className="py-2 pr-4">Output</th>
                    </tr>
                  </thead>
                  <tbody>
                    {run.results.map((item) => (
                      <tr key={item.model_id} className="border-b border-foreground/10 align-top">
                        <td className="py-3 pr-4 font-bold">{item.label}</td>
                        <td className="py-3 pr-4">{item.ok ? formatPct(item.metrics?.wer) : "failed"}</td>
                        <td className="py-3 pr-4 text-[#b91c1c]">{item.metrics?.missed_entities?.join(", ") || "-"}</td>
                        <td className="py-3 pr-4 theme-muted">{item.ok ? item.transcript : item.error}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          ))}
        </div>
      </div>
    </section>
  );
}

function PieChart({ items }: { items: Aggregate[] }) {
  const scored = items.map((item) => ({
    ...item,
    score: Math.max(0.01, 1 - Math.min(item.wer ?? 1, 1)),
  }));
  const total = scored.reduce((sum, item) => sum + item.score, 0) || 1;
  const colors = ["#a78bfa", "#fbbf24", "#60a5fa", "#34d399", "#f87171", "#c084fc"];
  const gradient = scored
    .map((item, index) => {
      const start = scored.slice(0, index).reduce((sum, previous) => sum + (previous.score / total) * 100, 0);
      const end = start + (item.score / total) * 100;
      return `${colors[index % colors.length]} ${start}% ${end}%`;
    })
    .join(", ");

  return (
    <div className="mt-5 grid gap-5 sm:grid-cols-[180px_1fr] sm:items-center">
      <div className="pie-chart" style={{ background: `conic-gradient(${gradient})` }} />
      <div className="space-y-3">
        {scored.map((item, index) => (
          <div key={item.model_id} className="legend-row">
            <span className="legend-dot" style={{ background: colors[index % colors.length] }} />
            <span className="font-bold">{item.label}</span>
            <span className="ml-auto theme-muted">{formatPct(item.score / total)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function LatencyChart({ items }: { items: Aggregate[] }) {
  const maxLatency = Math.max(...items.map((item) => item.latency_ms ?? 0), 1);
  return (
    <div className="mt-5 space-y-4">
      {items.map((item) => {
        const latency = item.latency_ms ?? 0;
        return (
          <div key={item.model_id} className="latency-row">
            <div className="flex items-center justify-between gap-4">
              <span className="font-bold">{item.label}</span>
              <span className="theme-muted">{latency ? `${latency} ms` : "n/a"}</span>
            </div>
            <div className="latency-track">
              <div className="latency-fill" style={{ width: `${(latency / maxLatency) * 100}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
