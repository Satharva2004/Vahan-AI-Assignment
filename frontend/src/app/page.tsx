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
  hallucination_rate: number | null;
  latency_ms: number | null;
  rtf: number | null;
  failures: number;
};

type RunResult = {
  model_id: string;
  provider: string;
  label: string;
  transcript: string;
  latency_ms: number;
  rtf?: number | null;
  realtime_status?: string | null;
  ok: boolean;
  error?: string;
  metrics?: {
    wer: number;
    cer: number;
    entity_f1: number;
    entity_recall: number;
    hallucination_rate: number;
    missed_entities: string[];
    inserted_words?: number;
  };
};

type Run = {
  file_name: string;
  reference: string;
  ground_truth?: string;
  duration_seconds?: number | null;
  entities?: string;
  condition?: string;
  metadata?: Record<string, string>;
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
  audioUrl?: string;
  fileName?: string;
  previewUrl?: string;
  reference: string;
  entities: string;
  condition: string;
  source?: string;
  durationSeconds?: number | null;
  metadata?: Record<string, string>;
};

type SampleDefinition = {
  id: string;
  title: string;
  language: string;
  speaker: string;
  audioUrl: string;
  fileName: string;
  condition: string;
  entities: string;
  reference: string;
  metadata: Record<string, string>;
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
const conditionOptions = ["Quiet", "Traffic", "Phone call", "Rushed", "Whispered", "Noisy room", "Hinglish", "Hindi", "Kannada", "Public sample"];
const modelIcons: Record<string, string> = {
  deepgram: "https://www.google.com/s2/favicons?domain=deepgram.com&sz=64",
  sarvam: "https://www.google.com/s2/favicons?domain=sarvam.ai&sz=64",
  assemblyai: "https://www.google.com/s2/favicons?domain=assemblyai.com&sz=64",
  openai: "https://www.google.com/s2/favicons?domain=openai.com&sz=64",
  google: "https://www.google.com/s2/favicons?domain=cloud.google.com&sz=64",
};
const modelColors = ["#e91e63", "#34a853", "#ff8a00", "#2f80ed", "#8b5cf6", "#009688", "#fbbf24"];

const defaultModels: ModelSpec[] = [
  { id: "deepgram-nova-3", provider: "deepgram", label: "Deepgram Nova-3", rationale: "" },
  { id: "sarvam-saaras-v3-transcribe", provider: "sarvam", label: "Sarvam Saaras v3", rationale: "" },
  { id: "sarvam-saaras-v3-codemix", provider: "sarvam", label: "Sarvam Saaras v3 Codemix", rationale: "" },
  { id: "assemblyai-best", provider: "assemblyai", label: "AssemblyAI Best", rationale: "" },
  { id: "google-stt-long", provider: "google", label: "Google STT Latest Long", rationale: "" },
  { id: "openai-gpt-4o-mini-transcribe", provider: "openai", label: "OpenAI GPT-4o Mini Transcribe", rationale: "" },
];

const voiceOfIndiaSamples: SampleDefinition[] = [
  {
    id: "hindi-speaker-1",
    title: "General Conversation",
    language: "Hindi",
    speaker: "Speaker 1",
    audioUrl: "https://storage.googleapis.com/data_uplaod_hcb/data/Hindi/3/speaker1_audio.wav",
    fileName: "voice-of-india-hindi-speaker-1.wav",
    condition: "Hindi",
    entities: "Katni, Madhya Pradesh",
    reference: "",
    metadata: {
      age: "35",
      gender: "Male",
      education: "Post Graduate",
      income: "INR 0 - 3 Lakhs",
      current_district: "Katni, Madhya Pradesh",
      mother_tongue: "Hindi",
      childhood_district: "Katni, Madhya Pradesh",
      dataset: "Voice of India conversational Hindi",
    },
  },
  {
    id: "hindi-speaker-2",
    title: "General Conversation",
    language: "Hindi",
    speaker: "Speaker 2",
    audioUrl: "https://storage.googleapis.com/data_uplaod_hcb/data/Hindi/3/speaker2_audio.wav",
    fileName: "voice-of-india-hindi-speaker-2.wav",
    condition: "Hindi",
    entities: "Kanpur Nagar",
    reference: "",
    metadata: {
      age: "21",
      gender: "Male",
      education: "12th Pass",
      income: "INR 0 - 3 Lakhs",
      current_district: "Kanpur Nagar",
      mother_tongue: "Hindi",
      childhood_district: "Kanpur Nagar",
      dataset: "Voice of India conversational Hindi",
    },
  },
  {
    id: "kannada-speaker-1",
    title: "Conversational Stem",
    language: "Kannada",
    speaker: "Speaker 1",
    audioUrl: "https://storage.googleapis.com/joshtalks-data-collection-248zy39c/transcription_merge/conversation_171049_left.wav",
    fileName: "voice-of-india-kannada-speaker-1.wav",
    condition: "Kannada",
    entities: "Uttar Kannada, Karnataka",
    reference: "",
    metadata: {
      age: "24",
      gender: "Female",
      education: "Graduate",
      income: "0 - 3 Lakhs",
      current_district: "Uttar Kannada, Karnataka",
      mother_tongue: "Kannada",
      childhood_district: "Uttar Kannada",
      dataset: "Voice of India conversational Kannada",
    },
  },
  {
    id: "kannada-speaker-2",
    title: "Conversational Stem",
    language: "Kannada",
    speaker: "Speaker 2",
    audioUrl: "https://storage.googleapis.com/joshtalks-data-collection-248zy39c/transcription_merge/conversation_171049_right.wav",
    fileName: "voice-of-india-kannada-speaker-2.wav",
    condition: "Kannada",
    entities: "Bangalore Urban, Karnataka",
    reference: "",
    metadata: {
      age: "48",
      gender: "Female",
      education: "Post Graduate",
      income: "0 - 3 Lakhs",
      current_district: "Bangalore Urban, Karnataka",
      mother_tongue: "Kannada",
      childhood_district: "Bangalore Urban",
      dataset: "Voice of India conversational Kannada",
    },
  },
];

function newRow(index: number): RecordingRow {
  return { id: `recording-${index}`, reference: "", entities: "", condition: "Quiet" };
}

function formatPct(value: number | null | undefined) {
  if (value == null) return "n/a";
  return `${Math.round(value * 1000) / 10}%`;
}

function formatRtf(value: number | null | undefined) {
  if (value == null) return "n/a";
  return `${Math.round(value * 100) / 100}x`;
}

function realtimeLabel(status: string | null | undefined) {
  if (status === "faster-than-real-time") return "Faster than real-time";
  if (status === "real-time") return "Real-time";
  if (status === "slower-than-real-time") return "Slower than real-time";
  return "n/a";
}

function readAudioDuration(src?: string) {
  return new Promise<number | null>((resolve) => {
    if (!src || typeof Audio === "undefined") {
      resolve(null);
      return;
    }
    const audio = new Audio();
    audio.preload = "metadata";
    audio.onloadedmetadata = () => {
      resolve(Number.isFinite(audio.duration) ? Math.round(audio.duration * 1000) / 1000 : null);
    };
    audio.onerror = () => resolve(null);
    audio.src = src;
  });
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
    const previewUrl = file ? URL.createObjectURL(file) : undefined;
    updateRow(id, {
      file,
      audioUrl: undefined,
      fileName: file?.name,
      previewUrl,
      durationSeconds: null,
      source: undefined,
      metadata: undefined,
    });
    if (previewUrl) {
      readAudioDuration(previewUrl).then((durationSeconds) => updateRow(id, { durationSeconds }));
    }
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

  function removeRecording(id: string) {
    setRows((current) => {
      const next = current.filter((row) => row.id !== id);
      return next.length ? next : [newRow(0)];
    });
  }

  function addSample(sample: SampleDefinition) {
    setRows((current) => {
      if (current.length >= 20) return current;
      const emptyIndex = current.findIndex((row) => !row.file && !row.audioUrl && !row.reference.trim() && !row.entities.trim());
      const sampleRow: RecordingRow = {
        id: `sample-${sample.id}-${Date.now()}`,
        audioUrl: sample.audioUrl,
        fileName: sample.fileName,
        previewUrl: sample.audioUrl,
        reference: sample.reference,
        entities: sample.entities,
        condition: sample.condition,
        source: `${sample.language} ${sample.speaker}`,
        durationSeconds: null,
        metadata: sample.metadata,
      };
      if (emptyIndex === -1) return [...current, sampleRow];
      return current.map((row, index) => (index === emptyIndex ? sampleRow : row));
    });
  }

  function selectModel(modelId: string) {
    setModelToAdd(modelId);
    if (modelId && modelId !== baselineModelId && !selected.includes(modelId)) {
      setSelected((current) => [...current, modelId]);
    }
  }

  async function ensureDurations(targetRows: RecordingRow[]) {
    const entries = await Promise.all(
      targetRows.map(async (row) => {
        if (row.durationSeconds) return [row.id, row.durationSeconds] as const;
        if (!(row.previewUrl || row.audioUrl)) return [row.id, null] as const;
        const durationSeconds = await readAudioDuration(row.previewUrl ?? row.audioUrl);
        if (durationSeconds) updateRow(row.id, { durationSeconds });
        return [row.id, durationSeconds] as const;
      }),
    );
    return new Map(entries);
  }

  async function runBenchmark(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    setProgress([]);

    try {
      const usableUploads = rows.filter((row) => row.file && row.reference.trim());
      const usableSamples = rows.filter((row) => row.audioUrl);
      if (!usableUploads.length && !usableSamples.length) throw new Error("Upload audio or choose a public sample.");
      const durationById = await ensureDurations([...usableUploads, ...usableSamples]);
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

        try {
          if (usableUploads.length) {
            const form = new FormData();
            usableUploads.forEach((row) => form.append("files", row.file as File));
            form.append("references", JSON.stringify(usableUploads.map((row) => row.reference)));
            form.append("entities", JSON.stringify(usableUploads.map((row) => row.entities)));
            form.append("conditions", JSON.stringify(usableUploads.map((row) => row.condition)));
            form.append("durations", JSON.stringify(usableUploads.map((row) => durationById.get(row.id) ?? null)));
            form.append("model_ids", JSON.stringify([modelId]));

            const response = await fetch(`${apiBase}/benchmark`, { method: "POST", body: form });
            const payload = (await response.json()) as BenchmarkResponse;
            if (!response.ok || !payload.ok) throw new Error(payload.error ?? `${label} failed.`);
            merged = mergeBenchmarkResults(merged, payload);
            setResult(merged);
          }

          if (usableSamples.length) {
            const response = await fetch(`${apiBase}/benchmark-urls`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                model_ids: [modelId],
                samples: usableSamples.map((row) => ({
                  audio_url: row.audioUrl,
                  file_name: row.fileName,
                  reference: row.reference,
                  entities: row.entities,
                  condition: row.condition,
                  duration_seconds: durationById.get(row.id),
                  metadata: row.metadata,
                })),
              }),
            });
            const payload = (await response.json()) as BenchmarkResponse;
            if (!response.ok || !payload.ok) throw new Error(payload.error ?? `${label} failed.`);
            merged = mergeBenchmarkResults(merged, payload);
            setResult(merged);
          }

          const elapsedMs = Math.round(performance.now() - started);
          setProgress((current) => current.map((item) => item.modelId === modelId ? { ...item, status: "done", elapsedMs } : item));
        } catch (modelError) {
          const elapsedMs = Math.round(performance.now() - started);
          const message = modelError instanceof Error ? modelError.message : `${label} failed.`;
          setProgress((current) => current.map((item) => item.modelId === modelId ? { ...item, status: "failed", elapsedMs, error: message } : item));
        }
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
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
          <p className="font-display text-3xl tracking-tight">AI ASR Lab</p>
          <nav className="hidden items-center gap-10 text-base font-bold md:flex">
            <span>Upload</span>
            <span>Compare</span>
            <span>Report</span>
            <button type="button" className="button-secondary h-14 px-5">Assignment</button>
          </nav>
        </div>
      </header>

      <section className="mx-auto max-w-4xl px-6 pb-8 pt-6 text-center">
        <h1 className="font-display text-3xl leading-tight md:text-4xl">Automatic Speech Recognition Benchmarking</h1>
        <p className="mx-auto mt-3 max-w-2xl text-base leading-6 theme-muted">
          Upload audio, add the ground truth, and compare ASR models.
        </p>
      </section>

      <form onSubmit={runBenchmark} className="mx-auto max-w-7xl px-6 pb-12">
        <section className="assignment-shell relative overflow-hidden">
          {loading ? <RunAnimation /> : null}

          <div className="mb-5 flex flex-col justify-between gap-5 md:flex-row md:items-start">
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
                    <ModelIcon provider={model.provider} label={model.label} />
                    {model.id === baselineModelId ? `${model.label} baseline` : model.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="workspace-grid">
            <SampleShelf samples={voiceOfIndiaSamples} onAdd={addSample} disabled={rows.length >= 20 || loading} />

            <div className="active-stack">
              <div className="active-header">
                <div>
                  <p className="text-sm font-bold theme-muted">Active audio</p>
                  <h3 className="font-display text-xl">{rows.length} item{rows.length === 1 ? "" : "s"}</h3>
                </div>
                <button type="button" className="button-secondary compact-button" onClick={addRecording} disabled={rows.length >= 20 || loading}>Add file</button>
              </div>

              {progress.length ? <ProgressPanel progress={progress} result={result} /> : null}

              <div className="recording-list">
                {rows.map((row, index) => (
                  <div key={row.id} className="recording-item">
                    <div className="recording-topline">
                      <div className="min-w-0">
                        <p className="font-display text-lg">Audio {index + 1}</p>
                        <p className="truncate text-sm theme-muted">{row.file?.name ?? row.fileName ?? "No audio selected"}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        {row.source ? (
                          <span className="source-chip metadata-trigger">
                            {row.source}
                            {row.metadata ? <MetadataPopover metadata={row.metadata} /> : null}
                          </span>
                        ) : null}
                        <button type="button" className="icon-button danger" onClick={() => removeRecording(row.id)} aria-label={`Remove audio ${index + 1}`}>X</button>
                      </div>
                    </div>

                    <div className="recording-compact-grid">
                      <div className={`compact-upload ${row.file || row.audioUrl ? "compact-upload-ready" : ""}`}>
                        <label className="compact-upload-button">
                          <input type="file" accept="audio/*,video/mp4,video/webm" onChange={(event) => updateFile(row.id, event)} />
                          <UploadGlyph />
                          <span>{row.file || row.audioUrl ? "Replace" : "Browse"}</span>
                        </label>

                        {recordingRowId === row.id ? (
                          <button type="button" className="record-button recording" onClick={stopRecording}>Stop {recordingSeconds}s</button>
                        ) : (
                          <button type="button" className="record-button" onClick={() => startRecording(row.id)} disabled={Boolean(recordingRowId)}>Record</button>
                        )}

                        {row.previewUrl || row.audioUrl ? <audio className="audio-preview compact-audio" src={row.previewUrl ?? row.audioUrl} controls /> : null}
                      </div>

                      <label className="compact-reference">
                        <span className="mb-1 block text-xs font-bold theme-muted">Ground truth</span>
                        <textarea
                          value={row.reference}
                          onChange={(event) => updateRow(row.id, { reference: event.target.value })}
                          placeholder="Type what is spoken"
                          className="field compact-textarea resize-y"
                        />
                      </label>

                      <label>
                        <span className="mb-1 block text-xs font-bold theme-muted">Entities</span>
                        <input
                          value={row.entities}
                          onChange={(event) => updateRow(row.id, { entities: event.target.value })}
                          placeholder="Places, names"
                          className="field compact-field"
                        />
                      </label>

                      <div>
                        <span className="mb-1 block text-xs font-bold theme-muted">Condition</span>
                        <div className="condition-strip">
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
                ))}
              </div>

              {error ? <p className="text-sm font-bold text-[#b91c1c]">{error}</p> : null}
            </div>
          </div>
        </section>
      </form>

      {result ? <Report result={result} readyForAnalysis={!loading && progress.every((item) => item.status === "done" || item.status === "failed")} /> : null}
    </main>
  );
}

function mergeBenchmarkResults(current: BenchmarkResponse | null, next: BenchmarkResponse): BenchmarkResponse {
  const runs = current ? mergeRuns(current.runs, next.runs) : next.runs;
  return {
    ...(current ?? next),
    aggregate: buildAggregate(runs),
    runs,
    methodology: next.methodology ?? current?.methodology ?? {},
    failure_analysis: buildFailureAnalysis(runs),
  };
}

function mergeRuns(currentRuns: Run[], nextRuns: Run[]) {
  const byName = new Map(currentRuns.map((run) => [run.file_name, { ...run, results: [...run.results] }]));
  nextRuns.forEach((run) => {
    const existing = byName.get(run.file_name);
    if (existing) {
      existing.results = [...existing.results, ...run.results];
      byName.set(run.file_name, existing);
    } else {
      byName.set(run.file_name, run);
    }
  });
  return Array.from(byName.values());
}

function buildAggregate(runs: Run[]): Aggregate[] {
  const byModel = new Map<string, { provider: string; label: string; results: RunResult[] }>();
  runs.forEach((run) => {
    run.results.forEach((result) => {
      const current = byModel.get(result.model_id) ?? { provider: result.provider, label: result.label, results: [] };
      current.results.push(result);
      byModel.set(result.model_id, current);
    });
  });

  return Array.from(byModel.entries())
    .map(([modelId, item]) => {
      const successes = item.results.filter((result) => result.ok);
      const metricResults = successes.filter((result) => result.metrics);
      return {
        model_id: modelId,
        provider: item.provider,
        label: item.label,
        recordings: successes.length,
        wer: avgMetric(metricResults, "wer"),
        cer: avgMetric(metricResults, "cer"),
        entity_f1: avgMetric(metricResults, "entity_f1"),
        entity_recall: avgMetric(metricResults, "entity_recall"),
        hallucination_rate: avgMetric(metricResults, "hallucination_rate"),
        latency_ms: successes.length ? Math.round(successes.reduce((sum, result) => sum + result.latency_ms, 0) / successes.length) : null,
        rtf: avgResult(successes, "rtf"),
        failures: item.results.length - successes.length,
      };
    })
    .sort((a, b) => (a.wer ?? 999) - (b.wer ?? 999));
}

function avgMetric(results: RunResult[], key: "wer" | "cer" | "entity_f1" | "entity_recall" | "hallucination_rate") {
  if (!results.length) return null;
  return Math.round((results.reduce((sum, result) => sum + (result.metrics?.[key] ?? 0), 0) / results.length) * 10000) / 10000;
}

function avgResult(results: RunResult[], key: "rtf") {
  const values = results.map((result) => result[key]).filter((value): value is number => value != null);
  if (!values.length) return null;
  return Math.round((values.reduce((sum, value) => sum + value, 0) / values.length) * 1000) / 1000;
}

function buildFailureAnalysis(runs: Run[]): FailureAnalysis {
  const byCondition: FailureAnalysis["by_condition"] = {};
  const byModel: FailureAnalysis["by_model"] = {};
  const byEntity: FailureAnalysis["by_entity"] = {};

  runs.forEach((run) => {
    const condition = run.condition || "Unlabeled";
    byCondition[condition] ??= { total: 0, failed_models: 0, missed_entities: 0 };
    byCondition[condition].total += 1;

    run.results.forEach((result) => {
      byModel[result.label] ??= { total: 0, failures: 0, missed_entities: 0, avg_wer: null };
      byModel[result.label].total += 1;
      if (!result.ok) {
        byModel[result.label].failures += 1;
        byCondition[condition].failed_models += 1;
        return;
      }
      if (!result.metrics) return;

      byModel[result.label].missed_entities += result.metrics.missed_entities.length;
      byModel[result.label].avg_wer = result.metrics.wer;
      byCondition[condition].missed_entities += result.metrics.missed_entities.length;
      result.metrics.missed_entities.forEach((entity) => {
        byEntity[entity] = (byEntity[entity] ?? 0) + 1;
      });
    });
  });

  return { by_condition: byCondition, by_model: byModel, by_entity: byEntity };
}

function SampleShelf({ samples, onAdd, disabled }: { samples: SampleDefinition[]; onAdd: (sample: SampleDefinition) => void; disabled: boolean }) {
  return (
    <div className="sample-shelf">
      <div>
        <p className="text-sm font-bold theme-muted">Public samples</p>
        <h3 className="font-display text-xl">Voice of India</h3>
      </div>
      <div className="sample-grid mt-4">
        {samples.map((sample) => (
          <button key={sample.id} type="button" className="sample-option" onClick={() => onAdd(sample)} disabled={disabled}>
            <span className="sample-language metadata-trigger">
              {sample.language}
              <MetadataPopover metadata={sample.metadata} />
            </span>
            <span className="font-bold">{sample.title}</span>
            <span className="text-sm theme-muted">{sample.speaker} / {sample.metadata.current_district}</span>
            <span className="sample-action">Add sample</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function MetadataPopover({ metadata }: { metadata: Record<string, string> }) {
  return (
    <span className="metadata-popover">
      {Object.entries(metadata).map(([key, value]) => (
        <span key={key} className="metadata-line">
          <span>{key.replaceAll("_", " ")}</span>
          <strong>{value}</strong>
        </span>
      ))}
    </span>
  );
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
                <p className="flex items-center gap-2 font-bold">
                  <ModelIcon provider={modelResult?.provider ?? ""} label={item.label} />
                  {item.label}
                </p>
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

function ModelIcon({ provider, label }: { provider: string; label: string }) {
  const src = modelIcons[provider];
  return src ? (
    // eslint-disable-next-line @next/next/no-img-element
    <img className="model-icon" src={src} alt={`${label} icon`} />
  ) : (
    <span className="model-icon-fallback">{label.slice(0, 1)}</span>
  );
}

function Report({ result, readyForAnalysis }: { result: BenchmarkResponse; readyForAnalysis: boolean }) {
  const best = result.aggregate.find((item) => item.failures === 0) ?? result.aggregate[0];
  const [analysis, setAnalysis] = useState<string>("");
  const [analysisSource, setAnalysisSource] = useState<string>("");
  const [overviewMetric, setOverviewMetric] = useState<"wer" | "accuracy">("wer");

  useEffect(() => {
    if (!readyForAnalysis) return;
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
  }, [result, readyForAnalysis]);

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
        <div className="markdown-summary mt-4 text-sm leading-6 theme-muted">
          {readyForAnalysis ? renderMarkdown(analysis || "Generating analysis...") : <p>Waiting for all selected models to finish.</p>}
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="theme-panel lg:col-span-2">
          <div className="flex flex-col justify-between gap-3 md:flex-row md:items-center">
            <div>
              <p className="text-sm font-bold theme-muted">{overviewMetric === "wer" ? "WER (%)" : "Accuracy (%)"}</p>
              <h3 className="font-display text-xl">Benchmark overview</h3>
            </div>
            <div className="metric-switch" aria-label="Benchmark overview metric">
              <button
                type="button"
                className={overviewMetric === "wer" ? "metric-switch-active" : ""}
                onClick={() => setOverviewMetric("wer")}
              >
                WER
              </button>
              <button
                type="button"
                className={overviewMetric === "accuracy" ? "metric-switch-active" : ""}
                onClick={() => setOverviewMetric("accuracy")}
              >
                Accuracy
              </button>
            </div>
          </div>
          <Leaderboard items={result.aggregate} metric={overviewMetric} />
        </div>

        <div className="theme-panel">
          <h3 className="font-display text-xl">Latency by model</h3>
          <LatencyChart items={result.aggregate} />
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
                <th className="py-2 pr-4">Hallucination</th>
                <th className="py-2 pr-4">RTF</th>
                <th className="py-2 pr-4">Latency</th>
                <th className="py-2 pr-4">Status</th>
              </tr>
            </thead>
            <tbody>
              {result.aggregate.map((item, index) => (
                <tr key={item.model_id} className="border-b border-foreground/10 align-top">
                  <td className="py-3 pr-4 font-bold">{index + 1}</td>
                  <td className="py-3 pr-4 font-bold">
                    <span className="inline-flex items-center gap-2">
                      <ModelIcon provider={item.provider} label={item.label} />
                      {item.label}
                    </span>
                  </td>
                  <td className="py-3 pr-4">{formatPct(item.wer)}</td>
                  <td className="py-3 pr-4">{formatPct(item.cer)}</td>
                  <td className="py-3 pr-4">{formatPct(item.entity_recall)}</td>
                  <td className="py-3 pr-4">{formatPct(item.hallucination_rate)}</td>
                  <td className="py-3 pr-4">{formatRtf(item.rtf)}</td>
                  <td className="py-3 pr-4">{item.latency_ms ? `${item.latency_ms} ms` : "n/a"}</td>
                  <td className="py-3 pr-4">{item.failures ? `${item.failures} failed` : "OK"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="theme-panel">
        <h3 className="font-display text-xl">Ground truth vs model output</h3>
        <div className="mt-4 space-y-5">
          {result.runs.map((run, index) => (
            <details key={`${run.file_name}-${index}`} open={index === 0}>
              <summary className="cursor-pointer font-bold">{run.file_name || `File ${index + 1}`}</summary>
              <p className="mt-2 text-sm theme-muted">Ground truth: {run.ground_truth || run.reference || "Not provided; transcript and latency only."}</p>
              <p className="mt-1 text-sm theme-muted">Audio duration: {run.duration_seconds ? `${run.duration_seconds}s` : "unknown"}</p>
              <div className="mt-3 overflow-x-auto">
                <table className="w-full min-w-[720px] border-collapse text-left text-sm">
                  <thead>
                    <tr className="border-y-2 border-foreground theme-muted">
                      <th className="py-2 pr-4">Model</th>
                      <th className="py-2 pr-4">WER</th>
                      <th className="py-2 pr-4">Hallucination</th>
                      <th className="py-2 pr-4">RTF</th>
                      <th className="py-2 pr-4">Missed</th>
                      <th className="py-2 pr-4">Model output</th>
                    </tr>
                  </thead>
                  <tbody>
                    {run.results.map((item) => (
                      <tr key={item.model_id} className="border-b border-foreground/10 align-top">
                        <td className="py-3 pr-4 font-bold">
                          <span className="inline-flex items-center gap-2">
                            <ModelIcon provider={item.provider} label={item.label} />
                            {item.label}
                          </span>
                        </td>
                        <td className="py-3 pr-4">{item.ok ? formatPct(item.metrics?.wer) : "failed"}</td>
                        <td className="py-3 pr-4">{item.ok ? formatPct(item.metrics?.hallucination_rate) : "failed"}</td>
                        <td className="py-3 pr-4">
                          <span className="font-bold">{formatRtf(item.rtf)}</span>
                          <span className="block text-xs theme-muted">{realtimeLabel(item.realtime_status)}</span>
                        </td>
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

function Leaderboard({ items, metric }: { items: Aggregate[]; metric: "wer" | "accuracy" }) {
  const chartItems = [...items].sort((a, b) => {
    if (metric === "accuracy") return accuracyValue(b) - accuracyValue(a);
    return (a.wer ?? 999) - (b.wer ?? 999);
  });
  const maxWer = Math.max(...chartItems.map((item) => item.wer ?? 0), 0.01);
  return (
    <div className="leaderboard mt-5">
      {chartItems.map((item, index) => {
        const wer = item.wer ?? maxWer;
        const value = metric === "accuracy" ? accuracyValue(item) : wer;
        const width = metric === "accuracy" ? Math.max(8, value * 100) : Math.max(8, (wer / maxWer) * 100);
        return (
          <div key={item.model_id} className="leaderboard-row">
            <span className="rank-badge">{index + 1}</span>
            <ModelIcon provider={item.provider} label={item.label} />
            <span className="leaderboard-name">{item.label}</span>
            <div className="leaderboard-track">
              <div
                className="leaderboard-fill"
                style={{
                  width: `${width}%`,
                  background: modelColors[index % modelColors.length],
                }}
              />
            </div>
            <span className="leaderboard-value">{metric === "accuracy" ? formatPct(value) : formatPct(item.wer)}</span>
          </div>
        );
      })}
    </div>
  );
}

function accuracyValue(item: Aggregate) {
  if (item.wer == null) return 0;
  return Math.max(0, 1 - item.wer);
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

function renderMarkdown(markdown: string) {
  return markdown.split("\n").map((line, index) => {
    if (line.startsWith("## ")) {
      return <h4 key={index}>{line.slice(3)}</h4>;
    }
    if (line.startsWith("- ")) {
      return <li key={index}>{renderInlineMarkdown(line.slice(2))}</li>;
    }
    if (!line.trim()) {
      return <span key={index} className="block h-2" />;
    }
    return <p key={index}>{renderInlineMarkdown(line)}</p>;
  });
}

function renderInlineMarkdown(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    return <span key={index}>{part}</span>;
  });
}
