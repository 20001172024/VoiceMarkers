import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Contribution } from "../api/types";
import { featureName, formatValue } from "../format";

const TOP_N = 8;

/** Contribution in percentage points of probability, signed. */
const points = (c: number) => `${c >= 0 ? "+" : "−"}${Math.abs(c * 100).toFixed(1)} pts`;

/**
 * Diverging bars: how much each feature moved this prediction's probability
 * (SHAP), towards the Parkinson's group (right, orange) or the healthy group
 * (left, blue). Every bar is also labelled, and a table holds all values.
 */
export function ContributionChart({ contributions }: { contributions: Contribution[] }) {
  const top = contributions.slice(0, TOP_N).map((c) => ({ ...c, name: featureName(c.feature) }));
  const extent = Math.max(...top.map((c) => Math.abs(c.contribution)), 0.001) * 1.35;

  return (
    <figure className="space-y-3">
      <figcaption>
        <h3 className="text-lg font-semibold">What moved this result</h3>
        <p className="text-sm text-ink-2">
          The {TOP_N} features with the largest effect on this probability, in percentage points.{" "}
          <span className="text-parkinsons font-medium">Right</span>: towards the Parkinson&apos;s
          group; <span className="text-healthy font-medium">left</span>: towards the healthy group.
          This describes what the model does with your values, not a medical finding.
        </p>
      </figcaption>
      <div className="h-80 w-full" aria-hidden="true">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={top} layout="vertical" margin={{ left: 8, right: 56, top: 4, bottom: 4 }}>
            <XAxis type="number" domain={[-extent, extent]} hide />
            <YAxis
              type="category"
              dataKey="name"
              width={150}
              tickLine={false}
              axisLine={false}
              tick={{ fill: "var(--ink)", fontSize: 12 }}
            />
            <ReferenceLine x={0} stroke="var(--ink-2)" />
            <Tooltip
              cursor={{ fill: "var(--grid)" }}
              formatter={(v) => points(Number(v))}
              labelStyle={{ color: "var(--ink)" }}
              itemStyle={{ color: "var(--ink)" }}
              contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)" }}
            />
            <Bar dataKey="contribution" barSize={14} radius={4} isAnimationActive={false}>
              {top.map((c) => (
                <Cell
                  key={c.feature}
                  fill={c.contribution >= 0 ? "var(--parkinsons)" : "var(--healthy)"}
                />
              ))}
              <LabelList
                dataKey="contribution"
                position="right"
                formatter={(v: unknown) => points(Number(v))}
                style={{ fill: "var(--ink)", fontSize: 12 }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <details className="text-sm">
        <summary className="cursor-pointer text-accent">
          Show all {contributions.length} features as a table
        </summary>
        <table className="mt-2 w-full text-left tabular-nums">
          <caption className="sr-only">Feature contributions to the probability</caption>
          <thead className="text-ink-2">
            <tr>
              <th scope="col" className="py-1 font-normal">
                Feature
              </th>
              <th scope="col" className="py-1 font-normal">
                Your value
              </th>
              <th scope="col" className="py-1 font-normal">
                Effect
              </th>
            </tr>
          </thead>
          <tbody>
            {contributions.map((c) => (
              <tr key={c.feature} className="border-t border-line">
                <th scope="row" className="py-1 font-normal">
                  {featureName(c.feature)}
                </th>
                <td className="py-1">{formatValue(c.value)}</td>
                <td className="py-1">{points(c.contribution)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
