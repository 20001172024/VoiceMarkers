import { useRecorder } from "../hooks/useRecorder";
import { Waveform } from "./Waveform";

interface Props {
  onRecorded: (blob: Blob) => void;
  disabled?: boolean;
}

const button =
  "rounded-md px-4 py-2 font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-50";

export function RecordPanel({ onRecorded, disabled }: Props) {
  const { state, analyser, start, stop, reset, recordSeconds } = useRecorder();
  const busy = ["requesting", "countdown", "recording"].includes(state.status);

  let status = "";
  if (state.status === "requesting") status = "Waiting for microphone permission…";
  if (state.status === "countdown") status = `Get ready… ${state.remaining}`;
  if (state.status === "recording")
    status = `Recording: keep saying "aaah"… ${Math.ceil(recordSeconds - state.elapsed)} s left`;
  if (state.status === "recorded") status = "Recording finished. Listen back, then analyse it.";

  return (
    <div className="space-y-3">
      <p className="text-sm text-ink-2">
        Take a breath and hold a steady &quot;aaah&quot; at a comfortable pitch and loudness for{" "}
        {recordSeconds} seconds, in a quiet room.
      </p>
      <p aria-live="polite" className="min-h-6 font-medium">
        {status}
      </p>
      {state.status === "error" && (
        <p role="alert" className="rounded-md bg-error-bg p-3 text-sm text-error-ink">
          {state.message}
        </p>
      )}
      {(state.status === "countdown" || state.status === "recording") && (
        <Waveform analyser={analyser} />
      )}
      {state.status === "recorded" && (
        <audio controls src={state.url} className="w-full" aria-label="Your recording" />
      )}
      <div className="flex flex-wrap gap-2">
        {state.status === "recorded" ? (
          <>
            <button
              type="button"
              disabled={disabled}
              onClick={() => onRecorded(state.blob)}
              className={`${button} bg-accent text-surface`}
            >
              Analyse this recording
            </button>
            <button type="button" onClick={reset} className={`${button} border border-line`}>
              Record again
            </button>
          </>
        ) : busy ? (
          <button
            type="button"
            onClick={state.status === "recording" ? stop : reset}
            className={`${button} border border-line`}
          >
            {state.status === "recording" ? "Stop" : "Cancel"}
          </button>
        ) : (
          <button
            type="button"
            disabled={disabled}
            onClick={() => void start()}
            className={`${button} bg-accent text-surface`}
          >
            Start recording
          </button>
        )}
      </div>
    </div>
  );
}
