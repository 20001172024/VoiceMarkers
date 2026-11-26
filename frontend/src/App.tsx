import { PRIVACY_NOTICE, REPO_URL } from "./constants";
import { AboutPage } from "./pages/AboutPage";
import { AnalysePage } from "./pages/AnalysePage";
import { useRoute } from "./router";

const navLink = (active: boolean) =>
  `rounded-md px-3 py-1.5 text-sm font-medium focus-visible:outline-2 focus-visible:outline-accent ${
    active ? "bg-surface-2 text-ink" : "text-ink-2 hover:text-ink"
  }`;

export default function App() {
  const route = useRoute();
  return (
    <div className="min-h-screen bg-surface text-ink">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:rounded-md focus:bg-surface focus:p-2"
      >
        Skip to content
      </a>
      <header className="border-b border-line">
        <nav
          aria-label="Main"
          className="mx-auto flex max-w-3xl items-center justify-between gap-4 px-4 py-3"
        >
          <a href="#/" className="text-lg font-bold tracking-tight">
            VoiceMarkers
          </a>
          <div className="flex gap-1">
            <a
              href="#/"
              aria-current={route === "analyse" ? "page" : undefined}
              className={navLink(route === "analyse")}
            >
              Analyse
            </a>
            <a
              href="#/about"
              aria-current={route === "about" ? "page" : undefined}
              className={navLink(route === "about")}
            >
              About the model
            </a>
          </div>
        </nav>
      </header>
      <main id="main" className="mx-auto max-w-3xl px-4 py-8">
        {route === "about" ? <AboutPage /> : <AnalysePage />}
      </main>
      <footer className="border-t border-line">
        <div className="mx-auto max-w-3xl space-y-1 px-4 py-6 text-xs text-ink-2">
          <p>Research and education demo. Not a medical device. {PRIVACY_NOTICE}</p>
          <p>
            Data: UCI Parkinson&apos;s dataset (Little et al., 2007), CC BY 4.0.{" "}
            <a className="underline" href={REPO_URL}>
              Source code
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}
