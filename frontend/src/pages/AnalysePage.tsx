import { lazy, Suspense, useState } from "react";
import { ApiError, predictAudio } from "../api/client";
import type { PredictionResponse } from "../api/types";
import { Disclaimer } from "../components/Disclaimer";
import { PrivacyNotice } from "../components/PrivacyNotice";
import { RecordPanel } from "../components/RecordPanel";
import { MAX_UPLOAD_BYTES } from "../constants";

// The results view pulls in the charting library; load it only when needed.
const Results = lazy(() =>
  import("../components/Results").then((module) => ({ default: module.Results })),
);

export const SAMPLES = [
  { file: "synthetic_steady.wav", label: "Steady synthetic voice" },
  { file: "synthetic_rough.wav", label: "Rough synthetic voice" },
];

type Phase =
  | { kind: "input" }
  | { kind: "analysing"; source: string }
  | { kind: "result"; result: PredictionResponse }
  | { kind: "error"; message: string };

const tab =
  "rounded-md px-3 py-2 text-sm font-medium focus-visible:outline-2 focus-visible:outline-accent aria-selected:bg-surface-2 aria-selected:text-ink text-ink-2";

export function AnalysePage() {
  const [phase, setPhase] = useState<Phase>({ kind: "input" });
  const [mode, setMode] = useState<"record" | "upload" | "sample">("record");

  async function analyse(audio: Blob, source: string) {
    setPhase({ kind: "analysing", source });
    try {
      const result = await predictAudio(audio);
      setPhase({ kind: "result", result });
    } catch (error) {
      const message =
        error instanceof ApiError ? error.message : "Something went wrong. Please try again.";
      setPhase({ kind: "error", message });
    }
  }

  async function analyseSample(file: string, label: string) {
    setPhase({ kind: "analysing", source: label });
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}samples/${file}`);
      if (!response.ok) throw new Error("sample missing");
      const blob = await response.blob();
      await analyse(new Blob([blob], { type: "audio/wav" }), label);
    } catch {
      setPhase({ kind: "error", message: "Could not load the sample recording." });
    }
  }

  function onUpload(file: File | undefined) {
    if (!file) return;
    if (file.size > MAX_UPLOAD_BYTES) {
      setPhase({
        kind: "error",
        message: "That file is larger than 5 MB. Please use a 3–5 second recording.",
      });
      return;
    }
    void analyse(file, file.name);
  }

  const busy = phase.kind === "analysing";

  if (phase.kind === "result") {
    return (
      <div>
        <Suspense fallback={<p aria-live="polite">Preparing your results…</p>}>
          <Results result={phase.result} onReset={() => setPhase({ kind: "input" })} />
        </Suspense>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <section className="space-y-4">
        <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
          What can a sustained &quot;aaah&quot; tell a machine-learning model?
        </h1>
        <p className="max-w-prose text-ink-2">
          VoiceMarkers measures pitch, jitter, shimmer and voice clarity from a short recording and
          compares them with a public research dataset of people with and without Parkinson&apos;s
          disease. It is the interactive companion to a research paper, and an honest look at how
          far such models really go.
        </p>
        <Disclaimer />
      </section>

      <section
        aria-labelledby="input-heading"
        className="space-y-4 rounded-lg border border-line p-5"
      >
        <h2 id="input-heading" className="text-xl font-semibold">
          Analyse a voice
        </h2>
        <div
          role="tablist"
          aria-label="How to provide a recording"
          className="flex flex-wrap gap-1"
        >
          {(
            [
              ["record", "Record"],
              ["upload", "Upload a file"],
              ["sample", "Try a sample"],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              type="button"
              role="tab"
              id={`tab-${key}`}
              aria-selected={mode === key}
              aria-controls={`panel-${key}`}
              onClick={() => setMode(key)}
              className={tab}
            >
              {label}
            </button>
          ))}
        </div>

        <div role="tabpanel" id={`panel-${mode}`} aria-labelledby={`tab-${mode}`}>
          {mode === "record" && (
            <RecordPanel
              disabled={busy}
              onRecorded={(blob) => void analyse(blob, "your recording")}
            />
          )}
          {mode === "upload" && (
            <div className="space-y-2">
              <label htmlFor="upload" className="block text-sm text-ink-2">
                A WAV file of a sustained vowel, 1.5–10 seconds long (max 5 MB).
              </label>
              <input
                id="upload"
                type="file"
                accept="audio/*,.wav"
                disabled={busy}
                onChange={(e) => onUpload(e.target.files?.[0])}
                className="block w-full text-sm file:mr-3 file:rounded-md file:border-0 file:bg-accent file:px-4 file:py-2 file:font-medium file:text-surface"
              />
            </div>
          )}
          {mode === "sample" && (
            <div className="space-y-3">
              <p className="text-sm text-ink-2">
                Synthetic voices generated by a script with known amounts of jitter, shimmer and
                noise. They are not recordings of real people.
              </p>
              <div className="flex flex-wrap gap-2">
                {SAMPLES.map((s) => (
                  <button
                    key={s.file}
                    type="button"
                    disabled={busy}
                    onClick={() => void analyseSample(s.file, s.label)}
                    className="rounded-md bg-accent px-4 py-2 font-medium text-surface focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-50"
                  >
                    Analyse: {s.label}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div aria-live="polite" className="min-h-6">
          {phase.kind === "analysing" && <p className="font-medium">Analysing {phase.source}…</p>}
          {phase.kind === "error" && (
            <p role="alert" className="rounded-md bg-error-bg p-3 text-sm text-error-ink">
              {phase.message}
            </p>
          )}
        </div>
        <PrivacyNotice />
      </section>
    </div>
  );
}
