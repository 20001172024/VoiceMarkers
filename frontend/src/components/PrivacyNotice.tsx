import { PRIVACY_NOTICE } from "../constants";

export function PrivacyNotice() {
  return (
    <p role="note" aria-label="Privacy" className="text-sm text-gray-700">
      <strong>Privacy:</strong> {PRIVACY_NOTICE}
    </p>
  );
}
