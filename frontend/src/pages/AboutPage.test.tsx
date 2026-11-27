import { render, screen } from "@testing-library/react";
import summary from "../generated/evaluation_summary.json";
import { meanStd, pct } from "../format";
import { AboutPage } from "./AboutPage";

describe("AboutPage", () => {
  it("shows the paper's figure and the generated evaluation numbers", () => {
    render(<AboutPage />);
    expect(
      screen.getByText(new RegExp(pct(summary.paper.reported_svm_accuracy, 2))),
    ).toBeInTheDocument();
    const svmSubject = meanStd(summary.protocols.SVM.subject_tuned);
    expect(
      screen.getAllByText(new RegExp(svmSubject.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))).length,
    ).toBeGreaterThan(0);
    expect(screen.getByRole("heading", { name: "Limitations" })).toBeInTheDocument();
    expect(screen.getByRole("note", { name: "Disclaimer" })).toBeInTheDocument();
  });

  it("lists every model in the protocol comparison", () => {
    render(<AboutPage />);
    for (const model of Object.keys(summary.protocols)) {
      expect(screen.getAllByRole("rowheader", { name: model }).length).toBeGreaterThan(0);
    }
  });
});

describe("AboutPage citation", () => {
  it("names the paper and its authors", () => {
    render(<AboutPage />);
    expect(screen.getByText(/Voice-Based Detection of Parkinson's Disease/)).toBeInTheDocument();
    expect(screen.getByText(/S\. Arora, S\. Jha, V\. Tyagi and Y\. Bhushan/)).toBeInTheDocument();
  });
});
