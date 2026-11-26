import { useEffect, useRef } from "react";
import type { PredictionResponse } from "../api/types";
import { ContributionChart } from "./ContributionChart";
import { Disclaimer } from "./Disclaimer";
import { FeatureRanges } from "./FeatureRanges";
import { ProbabilityGauge } from "./ProbabilityGauge";

interface Props {
  result: PredictionResponse;
  onReset: () => void;
}

export function Results({ result, onReset }: Props) {
  const { prediction } = result;
  const order = result.contributions.map((c) => c.feature);
  const heading = useRef<HTMLHeadingElement>(null);

  // Move focus to the result so keyboard and screen-reader users land on it.
  useEffect(() => heading.current?.focus(), []);

  return (
    <section aria-labelledby="results-heading" className="space-y-8">
      <div className="space-y-4 rounded-lg border border-line bg-surface-2 p-5">
        <h2
          id="results-heading"
          ref={heading}
          tabIndex={-1}
          className="text-sm font-medium uppercase tracking-wide text-ink-2 outline-none"
        >
          Result
        </h2>
        <p className="text-xl font-semibold">
          <span
            aria-hidden="true"
            className={`mr-2 inline-block h-3 w-3 rounded-full ${
              prediction.label === "parkinsons_like" ? "bg-parkinsons" : "bg-healthy"
            }`}
          />
          {prediction.label_text}
        </p>
        <ProbabilityGauge
          probability={prediction.probability}
          baseRate={prediction.base_rate}
          reference={prediction.reference_probability}
        />
        <p className="text-sm text-ink-2">
          The label comes from the model&apos;s own decision rule, which treats both groups equally.
          The probability leans towards Parkinson&apos;s for most voices because most training
          recordings were from people with Parkinson&apos;s, so compare it with the training average
          rather than with 50%.
        </p>
        {result.warnings.length > 0 && (
          <ul className="list-disc space-y-1 pl-5 text-sm text-warn-ink" aria-label="Warnings">
            {result.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        )}
      </div>

      <ContributionChart contributions={result.contributions} />
      <FeatureRanges ranges={result.feature_ranges} order={order} />

      <Disclaimer text={result.disclaimer} />
      <p className="text-xs text-ink-2">
        Model: {result.model.type}, version {result.model.version}. {result.privacy}
      </p>
      <button
        type="button"
        onClick={onReset}
        className="rounded-md border border-line px-4 py-2 font-medium hover:bg-surface-2 focus-visible:outline-2 focus-visible:outline-accent"
      >
        Try another recording
      </button>
    </section>
  );
}
