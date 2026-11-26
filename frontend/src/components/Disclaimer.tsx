import { DISCLAIMER } from "../constants";

export function Disclaimer({ text = DISCLAIMER }: { text?: string }) {
  return (
    <aside
      role="note"
      aria-label="Disclaimer"
      className="rounded-md border border-warn-line bg-warn-bg p-3 text-sm text-warn-ink"
    >
      <strong>Not a medical tool.</strong> {text}
    </aside>
  );
}
