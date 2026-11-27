import summary from "../generated/evaluation_summary.json";
import { Disclaimer } from "../components/Disclaimer";
import { EVALUATION_URL, PAPER_AUTHORS, PAPER_TITLE, PAPER_URL, PAPER_VENUE } from "../constants";
import { meanStd, pct } from "../format";

// Every number on this page comes from evaluation_summary.json, which
// research/report.py generates from the evaluation runs.

const PROTOCOLS = [
  ["row_default", "Row-level (paper setup)"],
  ["subject_default", "Subject-wise"],
  ["subject_tuned", "Subject-wise, tuned"],
] as const;

function Bar({ value }: { value: number }) {
  // 0.4-1.0 balanced-accuracy scale; 0.5 = chance.
  const width = Math.max(0, Math.min(1, (value - 0.4) / 0.6)) * 100;
  return (
    <span
      aria-hidden="true"
      className="ml-2 inline-block h-2 w-20 rounded bg-surface-2 align-middle"
    >
      <span className="block h-2 rounded bg-accent" style={{ width: `${width}%` }} />
    </span>
  );
}

export function AboutPage() {
  const { dataset: d, paper, deployed, reproducible } = summary;
  const svmRow = summary.protocols.SVM.row_default.mean;
  const svmSubject = summary.protocols.SVM.subject_default.mean;

  return (
    <article className="space-y-8">
      <header className="space-y-3">
        <h1 className="text-3xl font-bold tracking-tight">About the model</h1>
        <p className="max-w-prose text-ink-2">
          The paper{" "}
          <cite>
            {PAPER_URL ? (
              <a className="text-accent underline" href={PAPER_URL}>
                {PAPER_TITLE}
              </a>
            ) : (
              PAPER_TITLE
            )}
          </cite>{" "}
          ({PAPER_AUTHORS}, {PAPER_VENUE}) reported {pct(paper.reported_svm_accuracy, 2)} accuracy
          for an SVM on the UCI Parkinson&apos;s voice dataset. This page shows what happens when
          the same models are evaluated more strictly.
        </p>
      </header>

      <section className="space-y-2">
        <h2 className="text-xl font-semibold">The data</h2>
        <p className="max-w-prose">
          {d.n_recordings} recordings of a sustained vowel, labelled with {d.n_subject_ids} subject
          IDs: {d.n_parkinsons_subjects} with Parkinson&apos;s ({d.n_parkinsons} recordings) and{" "}
          {d.n_healthy_subjects} without ({d.n_healthy} recordings). The dataset&apos;s description
          says 31 people, so two IDs may belong to the same person. Always guessing
          &quot;Parkinson&apos;s&quot; is right {pct(summary.baseline_accuracy.mean)} of the time,
          so accuracy alone flatters any model. The numbers below use{" "}
          <strong>balanced accuracy</strong>, the average of the hit rates on both groups, where 50%
          is chance.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Why the headline number drops</h2>
        <p className="max-w-prose">
          Each person gave about six recordings. A random split of recordings puts the same person
          in both the training and the test set, so a model can score well by recognising the{" "}
          <em>voice</em>, not the disease. Keeping each person entirely on one side (a subject-wise
          split) is the honest test for a new person: SVM&apos;s balanced accuracy falls from{" "}
          {pct(svmRow)} to {pct(svmSubject)}. Over 200 random splits, the paper&apos;s setup gives{" "}
          {meanStd(paper.svm_accuracy_200_splits)} accuracy, so the published figure was a typical
          result of that setup.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[32rem] text-left text-sm tabular-nums">
            <caption className="mb-2 text-left text-ink-2">
              Balanced accuracy, mean ± std over {summary.n_outer_folds} test folds
            </caption>
            <thead className="text-ink-2">
              <tr>
                <th scope="col" className="py-1 font-normal">
                  Model
                </th>
                {PROTOCOLS.map(([, label]) => (
                  <th key={label} scope="col" className="py-1 font-normal">
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(summary.protocols).map(([model, byProtocol]) => (
                <tr key={model} className="border-t border-line">
                  <th scope="row" className="py-2 font-normal">
                    {model}
                  </th>
                  {PROTOCOLS.map(([key]) => (
                    <td key={key} className="py-2 whitespace-nowrap">
                      {meanStd(byProtocol[key])}
                      <Bar value={byProtocol[key].mean} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {reproducible && (
        <section className="space-y-3">
          <h2 className="text-xl font-semibold">Features the app can measure</h2>
          <p className="max-w-prose">
            The dataset has 22 features, but only {reproducible.n_features} (pitch, jitter, shimmer
            and harmonics-to-noise ratio) can be computed from a new recording with Praat. The
            others ({reproducible.dropped.join(", ")}) came from the original authors&apos; own
            code, so the deployed model does without them.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[24rem] text-left text-sm tabular-nums">
              <caption className="mb-2 text-left text-ink-2">
                Subject-wise balanced accuracy, tuned
              </caption>
              <thead className="text-ink-2">
                <tr>
                  <th scope="col" className="py-1 font-normal">
                    Model
                  </th>
                  <th scope="col" className="py-1 font-normal">
                    All 22 features
                  </th>
                  <th scope="col" className="py-1 font-normal">
                    {reproducible.n_features} measurable features
                  </th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(reproducible.balanced_accuracy).map(([model, v]) => (
                  <tr key={model} className="border-t border-line">
                    <th scope="row" className="py-2 font-normal">
                      {model}
                    </th>
                    <td className="py-2">{meanStd(v.all)}</td>
                    <td className="py-2">{meanStd(v.reproducible)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {deployed && (
        <section className="space-y-2">
          <h2 className="text-xl font-semibold">The model behind this app</h2>
          <p className="max-w-prose">
            {deployed.model_type} (version {deployed.model_version}), evaluated subject-wise on the{" "}
            {reproducible?.n_features} measurable features:
          </p>
          <dl className="grid max-w-md grid-cols-2 gap-x-4 gap-y-1 text-sm tabular-nums">
            <dt className="text-ink-2">Balanced accuracy</dt>
            <dd>{meanStd(deployed.metrics.balanced_accuracy)}</dd>
            <dt className="text-ink-2">Parkinson&apos;s recordings recognised</dt>
            <dd>{meanStd(deployed.metrics.recall)}</dd>
            <dt className="text-ink-2">Healthy recordings recognised</dt>
            <dd>{meanStd(deployed.metrics.specificity)}</dd>
            <dt className="text-ink-2">ROC-AUC</dt>
            <dd>{meanStd(deployed.metrics.roc_auc)}</dd>
          </dl>
        </section>
      )}

      <section className="space-y-2">
        <h2 className="text-xl font-semibold">Limitations</h2>
        <ul className="max-w-prose list-disc space-y-1 pl-5">
          <li>
            Only about {d.n_subject_ids} people, {d.n_healthy_subjects} of them healthy: every
            estimate has a wide margin, shown by the ± values.
          </li>
          <li>
            The training recordings were made in a controlled setting. Your microphone, room and
            browser change the measurements, so predictions on new recordings are not reliable.
          </li>
          <li>
            This app computes features with Praat; the dataset used different software (MDVP). The
            values are comparable in range, not identical.
          </li>
          <li>
            Voice changes have many causes. Nothing here can diagnose or rule out any condition.
          </li>
        </ul>
        <p>
          <a className="text-accent underline" href={EVALUATION_URL}>
            Full evaluation report
          </a>
        </p>
      </section>

      <Disclaimer />
    </article>
  );
}
