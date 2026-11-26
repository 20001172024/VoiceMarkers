import { act, render, screen } from "@testing-library/react";
import App from "./App";
import { DISCLAIMER } from "./constants";

afterEach(() => {
  window.location.hash = "";
});

describe("App", () => {
  it("renders the analyse page with the disclaimer by default", () => {
    render(<App />);
    expect(screen.getByRole("link", { name: "Analyse" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("note", { name: "Disclaimer" })).toHaveTextContent(DISCLAIMER);
  });

  it("navigates to the About page via the hash", () => {
    render(<App />);
    act(() => {
      window.location.hash = "#/about";
      window.dispatchEvent(new HashChangeEvent("hashchange"));
    });
    expect(screen.getByRole("heading", { level: 1, name: "About the model" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "About the model" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  it("has a skip link to the main content", () => {
    render(<App />);
    expect(screen.getByRole("link", { name: "Skip to content" })).toHaveAttribute("href", "#main");
  });
});
