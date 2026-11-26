import { DISCLAIMER } from "../constants";

export function Disclaimer() {
  return (
    <aside
      role="note"
      aria-label="Disclaimer"
      className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900"
    >
      <strong>Disclaimer:</strong> {DISCLAIMER}
    </aside>
  );
}
