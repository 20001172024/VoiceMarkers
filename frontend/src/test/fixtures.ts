import type { PredictionResponse } from "../api/types";
import raw from "./prediction.fixture.json";

/** A real /predict/audio response for the synthetic steady sample. */
export const prediction = raw as PredictionResponse;

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
