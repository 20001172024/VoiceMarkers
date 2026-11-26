/** Formatting helpers and plain-language names for the UCI feature columns. */

export const pct = (x: number, digits = 0): string => `${(x * 100).toFixed(digits)}%`;

export const meanStd = (s: { mean: number; std: number }): string =>
  `${pct(s.mean, 1)} ± ${pct(s.std, 1)}`;

interface FeatureInfo {
  name: string;
  description: string;
}

export const FEATURES: Record<string, FeatureInfo> = {
  "MDVP:Fo(Hz)": { name: "Average pitch", description: "Mean vocal pitch, in Hz." },
  "MDVP:Fhi(Hz)": { name: "Highest pitch", description: "Maximum vocal pitch, in Hz." },
  "MDVP:Flo(Hz)": { name: "Lowest pitch", description: "Minimum vocal pitch, in Hz." },
  "MDVP:Jitter(%)": {
    name: "Jitter (local)",
    description: "Average change in period from one voice cycle to the next (as a fraction).",
  },
  "MDVP:Jitter(Abs)": {
    name: "Jitter (absolute)",
    description: "Average cycle-to-cycle period change, in seconds.",
  },
  "MDVP:RAP": { name: "Jitter (RAP)", description: "Period variation averaged over 3 cycles." },
  "MDVP:PPQ": { name: "Jitter (PPQ5)", description: "Period variation averaged over 5 cycles." },
  "Jitter:DDP": {
    name: "Jitter (DDP)",
    description: "Difference of differences of periods (3 × RAP).",
  },
  "MDVP:Shimmer": {
    name: "Shimmer (local)",
    description: "Average change in loudness from one voice cycle to the next.",
  },
  "MDVP:Shimmer(dB)": {
    name: "Shimmer (dB)",
    description: "Cycle-to-cycle loudness change, in dB.",
  },
  "Shimmer:APQ3": { name: "Shimmer (APQ3)", description: "Loudness variation over 3 cycles." },
  "Shimmer:APQ5": { name: "Shimmer (APQ5)", description: "Loudness variation over 5 cycles." },
  "MDVP:APQ": { name: "Shimmer (APQ11)", description: "Loudness variation over 11 cycles." },
  "Shimmer:DDA": {
    name: "Shimmer (DDA)",
    description: "Difference of differences of amplitudes (3 × APQ3).",
  },
  HNR: {
    name: "Harmonics-to-noise ratio",
    description: "How much of the sound is regular voice versus noise, in dB (higher = clearer).",
  },
};

export const featureName = (feature: string): string => FEATURES[feature]?.name ?? feature;

/** Show small and large numbers readably (jitter is ~0.005, pitch ~150). */
export function formatValue(x: number): string {
  if (x === 0) return "0";
  const abs = Math.abs(x);
  if (abs >= 100) return x.toFixed(0);
  if (abs >= 1) return x.toFixed(1);
  if (abs >= 0.001) return x.toFixed(4);
  return x.toExponential(1);
}
