import { fireEvent, render, screen, within } from "@testing-library/react";
import { Results } from "./Results";
import { prediction } from "../test/fixtures";

describe("Results", () => {
  it("shows the label, probability with its context, and the disclaimer", () => {
    render(<Results result={prediction} onReset={() => {}} />);
    expect(screen.getByText(prediction.prediction.label_text)).toBeInTheDocument();
    const gauge = screen.getByRole("img", { name: /Probability of the Parkinson's group/ });
    expect(gauge).toHaveAccessibleName(/75%/); // base rate is part of the description
    expect(screen.getByRole("note", { name: "Disclaimer" })).toHaveTextContent(
      prediction.disclaimer,
    );
    for (const warning of prediction.warnings)
      expect(screen.getByText(warning)).toBeInTheDocument();
  });

  it("lists every feature contribution in the table view", () => {
    render(<Results result={prediction} onReset={() => {}} />);
    const table = screen.getByRole("table", { name: /Feature contributions/ });
    expect(within(table).getAllByRole("row")).toHaveLength(prediction.contributions.length + 1);
  });

  it("shows six feature ranges, then all of them on request", async () => {
    render(<Results result={prediction} onReset={() => {}} />);
    const rows = () => screen.getAllByRole("img", { name: /your value/ });
    expect(rows()).toHaveLength(6);
    fireEvent.click(screen.getByRole("button", { name: /Show all 15 features/ }));
    expect(rows()).toHaveLength(15);
  });

  it("moves focus to the result heading", () => {
    render(<Results result={prediction} onReset={() => {}} />);
    expect(screen.getByRole("heading", { name: "Result" })).toHaveFocus();
  });

  it("calls onReset", async () => {
    const onReset = vi.fn();
    render(<Results result={prediction} onReset={onReset} />);
    fireEvent.click(screen.getByRole("button", { name: "Try another recording" }));
    expect(onReset).toHaveBeenCalled();
  });
});
