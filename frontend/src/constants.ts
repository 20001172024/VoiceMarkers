// Keep in sync with backend/app/constants.py. From M4 the UI shows the
// disclaimer returned by the API; this copy covers pages with no API response.
export const DISCLAIMER =
  "VoiceMarkers is a research and education demo. It is not a medical device and " +
  "does not diagnose, screen for or rule out Parkinson's disease or any other " +
  "condition. Predictions on new recordings are not reliable: microphones and " +
  "recording conditions differ from the training data. Do not use it to make " +
  "health decisions.";

export const PRIVACY_NOTICE =
  "Audio is processed in memory and is never stored, logged or sent to third parties.";

export const REPO_URL = "https://github.com/20001172024/VoiceMarkers";
export const EVALUATION_URL = `${REPO_URL}/blob/main/docs/evaluation.md`;
/** Link to the ICOIICS 2025 paper; leave empty until a public URL is available. */
export const PAPER_URL = "";
export const MAX_UPLOAD_BYTES = 5_000_000;
