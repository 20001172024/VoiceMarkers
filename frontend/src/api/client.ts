import type { ErrorBody, PredictionResponse } from "./types";

export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";

/** An error the person can act on: `message` is already written for them. */
export class ApiError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(
      "network_error",
      "Could not reach the analysis server. Check your connection and try again.",
    );
  }
  if (response.ok) return (await response.json()) as T;

  let body: ErrorBody | undefined;
  try {
    body = (await response.json()) as ErrorBody;
  } catch {
    body = undefined;
  }
  throw new ApiError(
    body?.error.code ?? "http_error",
    body?.error.message ?? `The server returned an error (${response.status}).`,
    response.status,
  );
}

/** Send a recording as the raw request body; it is analysed in memory and never stored. */
export function predictAudio(audio: Blob): Promise<PredictionResponse> {
  return request<PredictionResponse>("/predict/audio", {
    method: "POST",
    headers: { "Content-Type": audio.type || "application/octet-stream" },
    body: audio,
  });
}
