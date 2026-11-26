import { pct } from "../format";

interface Props {
  probability: number;
  baseRate: number;
  reference: number;
}

/**
 * A 0-100% scale with the prediction marked, plus the two numbers needed to read
 * it: the training base rate and the model's average over training recordings.
 * Without them a probability like 74% looks alarming, when it is below average.
 */
export function ProbabilityGauge({ probability, baseRate, reference }: Props) {
  const x = (p: number) => 2 + p * 96; // % across the track, leaving room for the marker
  const summary =
    `Probability of the Parkinson's group: ${pct(probability)}. ` +
    `${pct(baseRate)} of training recordings were from people with Parkinson's; ` +
    `the model's average on training recordings is ${pct(reference)}.`;

  return (
    <figure className="space-y-2">
      <figcaption className="flex flex-wrap items-baseline justify-between gap-x-4">
        <span className="text-sm text-ink-2">Probability of the Parkinson&apos;s group</span>
        <span className="text-3xl font-semibold tabular-nums">{pct(probability)}</span>
      </figcaption>
      <svg
        role="img"
        aria-label={summary}
        viewBox="0 0 100 14"
        className="h-10 w-full"
        preserveAspectRatio="none"
      >
        <rect x="2" y="5" width="96" height="4" rx="2" fill="var(--grid)" />
        <line
          x1={x(baseRate)}
          x2={x(baseRate)}
          y1="2"
          y2="12"
          stroke="var(--ink-2)"
          strokeWidth="0.4"
          strokeDasharray="1 0.8"
        />
        <line
          x1={x(reference)}
          x2={x(reference)}
          y1="2"
          y2="12"
          stroke="var(--ink-2)"
          strokeWidth="0.4"
        />
        <rect
          x={x(probability) - 1}
          y="1.5"
          width="2"
          height="11"
          rx="1"
          fill="var(--ink)"
          stroke="var(--surface)"
          strokeWidth="0.5"
        />
      </svg>
      <div className="flex justify-between text-xs text-ink-2 tabular-nums" aria-hidden="true">
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
      <dl className="grid gap-1 text-sm text-ink-2 sm:grid-cols-2">
        <div>
          <dt className="inline">┆ Training base rate: </dt>
          <dd className="inline tabular-nums text-ink">{pct(baseRate)}</dd>
          <span> of recordings were Parkinson&apos;s</span>
        </div>
        <div>
          <dt className="inline">│ Model&apos;s training average: </dt>
          <dd className="inline tabular-nums text-ink">{pct(reference)}</dd>
        </div>
      </dl>
    </figure>
  );
}
