import { useCallback, useEffect, useRef, useState } from "react";

export type RecorderState =
  | { status: "idle" }
  | { status: "requesting" }
  | { status: "countdown"; remaining: number }
  | { status: "recording"; elapsed: number }
  | { status: "recorded"; blob: Blob; url: string }
  | { status: "error"; message: string };

export interface RecorderOptions {
  countdownSeconds?: number;
  recordSeconds?: number;
}

/** Formats in order of preference; the backend decodes all of them. */
const MIME_TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];

function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) return undefined;
  return MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type));
}

/** Turn getUserMedia failures into something the person can act on. */
export function microphoneErrorMessage(error: unknown): string {
  const name = error instanceof DOMException || error instanceof Error ? error.name : "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Microphone access was blocked. Allow it in your browser's site settings, or upload a WAV file instead.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No microphone was found. Connect one, or upload a WAV file instead.";
    case "NotReadableError":
      return "Your microphone is being used by another app. Close it and try again.";
    default:
      return "Recording is not available in this browser. Please upload a WAV file instead.";
  }
}

/**
 * Record a short sustained vowel: permission -> countdown -> fixed-length recording.
 *
 * Exposes an AnalyserNode for a live waveform while recording. Audio stays in the
 * browser as a Blob until the person chooses to submit it.
 */
export function useRecorder({ countdownSeconds = 3, recordSeconds = 4 }: RecorderOptions = {}) {
  const [state, setState] = useState<RecorderState>({ status: "idle" });
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);

  const stream = useRef<MediaStream | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const timers = useRef<ReturnType<typeof setInterval>[]>([]);
  const url = useRef<string | null>(null);

  const clearTimers = () => {
    timers.current.forEach(clearInterval);
    timers.current = [];
  };

  /** Release the microphone and audio graph (the recorded Blob is kept). */
  const release = useCallback(() => {
    clearTimers();
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
    void audioContext.current?.close().catch(() => undefined);
    audioContext.current = null;
    setAnalyser(null);
  }, []);

  const stop = useCallback(() => {
    clearTimers();
    if (recorder.current?.state === "recording") recorder.current.stop();
  }, []);

  const beginRecording = useCallback(() => {
    const media = stream.current;
    if (!media) return;
    const mimeType = pickMimeType();
    const rec = new MediaRecorder(media, mimeType ? { mimeType } : undefined);
    const chunks: Blob[] = [];
    rec.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };
    rec.onstop = () => {
      release();
      const blob = new Blob(chunks, { type: rec.mimeType || mimeType || "audio/webm" });
      url.current = URL.createObjectURL(blob);
      setState({ status: "recorded", blob, url: url.current });
    };
    recorder.current = rec;
    rec.start();

    const started = Date.now();
    setState({ status: "recording", elapsed: 0 });
    timers.current.push(
      setInterval(() => {
        const elapsed = (Date.now() - started) / 1000;
        if (elapsed >= recordSeconds) stop();
        else setState({ status: "recording", elapsed });
      }, 100),
    );
  }, [recordSeconds, release, stop]);

  const start = useCallback(async () => {
    if (url.current) URL.revokeObjectURL(url.current);
    url.current = null;
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setState({ status: "error", message: microphoneErrorMessage(null) });
      return;
    }
    setState({ status: "requesting" });
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
    } catch (error) {
      setState({ status: "error", message: microphoneErrorMessage(error) });
      return;
    }

    // The waveform is optional: skip it where the Web Audio API is missing.
    if (typeof AudioContext !== "undefined") {
      const context = new AudioContext();
      const node = context.createAnalyser();
      node.fftSize = 2048;
      context.createMediaStreamSource(stream.current).connect(node);
      audioContext.current = context;
      setAnalyser(node);
    }

    let remaining = countdownSeconds;
    if (remaining <= 0) {
      beginRecording();
      return;
    }
    setState({ status: "countdown", remaining });
    timers.current.push(
      setInterval(() => {
        remaining -= 1;
        if (remaining > 0) {
          setState({ status: "countdown", remaining });
        } else {
          clearTimers();
          beginRecording();
        }
      }, 1000),
    );
  }, [beginRecording, countdownSeconds]);

  const reset = useCallback(() => {
    stop();
    release();
    if (url.current) URL.revokeObjectURL(url.current);
    url.current = null;
    setState({ status: "idle" });
  }, [release, stop]);

  // Leaving the page mid-recording must release the microphone.
  useEffect(
    () => () => {
      if (recorder.current?.state === "recording") {
        recorder.current.onstop = null;
        recorder.current.stop();
      }
      release();
      if (url.current) URL.revokeObjectURL(url.current);
    },
    [release],
  );

  return { state, analyser, start, stop, reset, recordSeconds };
}
