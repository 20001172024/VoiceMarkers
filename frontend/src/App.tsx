import { Disclaimer } from "./components/Disclaimer";
import { PrivacyNotice } from "./components/PrivacyNotice";

export default function App() {
  return (
    <main className="mx-auto max-w-3xl space-y-6 p-6">
      <header>
        <h1 className="text-3xl font-bold">VoiceMarkers</h1>
        <p className="text-gray-700">
          Voice-based Parkinson&apos;s detection with machine learning — a research demo.
        </p>
      </header>
      <Disclaimer />
      <PrivacyNotice />
      <p className="text-gray-600">Recording and analysis are coming in later milestones.</p>
    </main>
  );
}
