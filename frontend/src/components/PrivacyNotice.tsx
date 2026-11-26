import { PRIVACY_NOTICE } from "../constants";

export function PrivacyNotice() {
  return (
    <p role="note" aria-label="Privacy" className="text-sm text-ink-2">
      <strong className="text-ink">Privacy:</strong> {PRIVACY_NOTICE}
    </p>
  );
}
