import { render, screen } from "@testing-library/react";
import App from "./App";
import { DISCLAIMER, PRIVACY_NOTICE } from "./constants";

describe("App", () => {
  it("renders the title", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: "VoiceMarkers" })).toBeInTheDocument();
  });

  it("always shows the disclaimer", () => {
    render(<App />);
    expect(screen.getByRole("note", { name: "Disclaimer" })).toHaveTextContent(DISCLAIMER);
  });

  it("shows the privacy notice", () => {
    render(<App />);
    expect(screen.getByRole("note", { name: "Privacy" })).toHaveTextContent(PRIVACY_NOTICE);
  });
});
