import { useState } from "react";
import type { FeatureRange } from "../api/types";
import { FEATURES, featureName, formatValue } from "../format";

const DEFAULT_SHOWN = 6;

function RangeRow({ range }: { range: FeatureRange }) {
  const { healthy: h, parkinsons: p, value } = range;
  const lo = Math.min(h.p5, p.p5, value);
  const hi = Math.max(h.p95, p.p95, value);
  const pad = (hi - lo) * 0.06 || 1;
  const x = (v: number) => 4 + ((v - lo + pad) / (hi - lo + 2 * pad)) * 92;

  const band = (q: FeatureRange["healthy"], y: number, color: string) => (
    <g>
      <line
        x1={x(q.p5)}
        x2={x(q.p95)}
        y1={y}
        y2={y}
        stroke={color}
        strokeWidth="1.2"
        strokeLinecap="round"
      />
      <line
        x1={x(q.p25)}
        x2={x(q.p75)}
        y1={y}
        y2={y}
        stroke={color}
        strokeWidth="4"
        strokeLinecap="round"
      />
      <line
        x1={x(q.p50)}
        x2={x(q.p50)}
        y1={y - 3}
        y2={y + 3}
        stroke="var(--surface)"
        strokeWidth="0.8"
      />
    </g>
  );

  const label =
    `${featureName(range.feature)}: your value ${formatValue(value)}. ` +
    `Healthy group middle half ${formatValue(h.p25)} to ${formatValue(h.p75)}; ` +
    `Parkinson's group middle half ${formatValue(p.p25)} to ${formatValue(p.p75)}.`;

  return (
    <li className="grid grid-cols-[minmax(0,9rem)_1fr] items-center gap-2 sm:grid-cols-[12rem_1fr_5rem]">
      <span className="text-sm" title={FEATURES[range.feature]?.description}>
        {featureName(range.feature)}
        {!range.within_training_range && (
          <span className="ml-1 text-xs text-warn-ink">(outside training range)</span>
        )}
      </span>
      <svg
        role="img"
        aria-label={label}
        viewBox="0 0 100 20"
        className="h-8 w-full"
        preserveAspectRatio="none"
      >
        {band(h, 6, "var(--healthy)")}
        {band(p, 14, "var(--parkinsons)")}
        <line x1={x(value)} x2={x(value)} y1="1" y2="19" stroke="var(--ink)" strokeWidth="0.6" />
        <circle
          cx={x(value)}
          cy="10"
          r="1.8"
          fill="var(--ink)"
          stroke="var(--surface)"
          strokeWidth="0.6"
        />
      </svg>
      <span className="hidden text-right text-sm tabular-nums sm:block">{formatValue(value)}</span>
    </li>
  );
}

/**
 * Where each of your values falls relative to the two training groups.
 * Thin line = 5th-95th percentile, thick = middle half, white tick = median.
 */
export function FeatureRanges({ ranges, order }: { ranges: FeatureRange[]; order: string[] }) {
  const [showAll, setShowAll] = useState(false);
  const sorted = [...ranges].sort((a, b) => order.indexOf(a.feature) - order.indexOf(b.feature));
  const shown = showAll ? sorted : sorted.slice(0, DEFAULT_SHOWN);

  return (
    <figure className="space-y-3">
      <figcaption>
        <h3 className="text-lg font-semibold">Your values compared with the training data</h3>
        <p className="text-sm text-ink-2">
          Each row shows the <span className="font-medium text-healthy">healthy group</span> (top)
          and the <span className="font-medium text-parkinsons">Parkinson&apos;s group</span>{" "}
          (bottom): thin line 5th–95th percentile, thick line middle half. The dark marker is your
          value. The groups overlap a lot, which is why a single voice says little.
        </p>
      </figcaption>
      <ul className="space-y-2">
        {shown.map((r) => (
          <RangeRow key={r.feature} range={r} />
        ))}
      </ul>
      {sorted.length > DEFAULT_SHOWN && (
        <button
          type="button"
          onClick={() => setShowAll((v) => !v)}
          aria-expanded={showAll}
          className="text-sm text-accent underline-offset-2 hover:underline"
        >
          {showAll ? "Show fewer" : `Show all ${sorted.length} features`}
        </button>
      )}
      <details className="text-sm">
        <summary className="cursor-pointer text-accent">Show as a table</summary>
        <table className="mt-2 w-full text-left tabular-nums">
          <caption className="sr-only">Your values and the training groups' medians</caption>
          <thead className="text-ink-2">
            <tr>
              <th scope="col" className="py-1 font-normal">
                Feature
              </th>
              <th scope="col" className="py-1 font-normal">
                You
              </th>
              <th scope="col" className="py-1 font-normal">
                Healthy median
              </th>
              <th scope="col" className="py-1 font-normal">
                Parkinson&apos;s median
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => (
              <tr key={r.feature} className="border-t border-line">
                <th scope="row" className="py-1 font-normal">
                  {featureName(r.feature)}
                </th>
                <td className="py-1">{formatValue(r.value)}</td>
                <td className="py-1">{formatValue(r.healthy.p50)}</td>
                <td className="py-1">{formatValue(r.parkinsons.p50)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
