import { fireEvent, render, screen } from "@testing-library/react";
import { AnalysePage } from "./AnalysePage";
import { jsonResponse, prediction } from "../test/fixtures";

afterEach(() => vi.restoreAllMocks());

describe("AnalysePage", () => {
  it("shows the disclaimer and privacy notice before any prediction", () => {
    render(<AnalysePage />);
    expect(screen.getByRole("note", { name: "Disclaimer" })).toBeInTheDocument();
    expect(screen.getByRole("note", { name: "Privacy" })).toBeInTheDocument();
  });

  it("analyses a sample in one click and shows the result", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("RIFF", { status: 200 })) // the sample WAV
      .mockResolvedValueOnce(jsonResponse(prediction)); // the API
    render(<AnalysePage />);

    fireEvent.click(screen.getByRole("tab", { name: "Try a sample" }));
    fireEvent.click(screen.getByRole("button", { name: /Steady synthetic voice/ }));

    expect(await screen.findByText(prediction.prediction.label_text)).toBeInTheDocument();
    expect(String(fetchMock.mock.calls[0][0])).toMatch(/samples\/synthetic_steady\.wav$/);
    expect(String(fetchMock.mock.calls[1][0])).toMatch(/\/predict\/audio$/);
  });

  it("shows the server's message when a recording is rejected", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(
        { error: { code: "too_short", message: "Hold the 'aaah' for longer." }, disclaimer: "d" },
        422,
      ),
    );
    render(<AnalysePage />);
    fireEvent.click(screen.getByRole("tab", { name: "Upload a file" }));
    fireEvent.change(screen.getByLabelText(/WAV file/), {
      target: { files: [new File(["RIFF"], "a.wav", { type: "audio/wav" })] },
    });
    expect(await screen.findByRole("alert")).toHaveTextContent("Hold the 'aaah' for longer.");
  });

  it("refuses files over 5 MB before uploading", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    render(<AnalysePage />);
    fireEvent.click(screen.getByRole("tab", { name: "Upload a file" }));
    const big = new File([new Uint8Array(5_000_001)], "big.wav", { type: "audio/wav" });
    fireEvent.change(screen.getByLabelText(/WAV file/), { target: { files: [big] } });
    expect(screen.getByRole("alert")).toHaveTextContent(/larger than 5 MB/);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("tabs are real buttons (keyboard reachable) and switch panels", () => {
    render(<AnalysePage />);
    const upload = screen.getByRole("tab", { name: "Upload a file" });
    expect(upload.tagName).toBe("BUTTON");
    fireEvent.click(upload);
    expect(upload).toHaveAttribute("aria-selected", "true");
    expect(screen.getByLabelText(/WAV file/)).toBeInTheDocument();
  });
});
