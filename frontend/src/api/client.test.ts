import { ApiError, predictAudio } from "./client";
import { jsonResponse, prediction } from "../test/fixtures";

afterEach(() => vi.restoreAllMocks());

describe("predictAudio", () => {
  it("posts the raw audio with its content type", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(prediction));
    const audio = new Blob(["RIFF"], { type: "audio/webm;codecs=opus" });

    const result = await predictAudio(audio);

    expect(result.prediction.label).toBe(prediction.prediction.label);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/predict\/audio$/);
    expect(init?.method).toBe("POST");
    expect(init?.body).toBe(audio);
    expect((init?.headers as Record<string, string>)["Content-Type"]).toBe(
      "audio/webm;codecs=opus",
    );
  });

  it("surfaces the server's user-facing error message and code", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(
        { error: { code: "silent", message: "The recording is silent." }, disclaimer: "d" },
        422,
      ),
    );
    await expect(predictAudio(new Blob())).rejects.toMatchObject({
      code: "silent",
      status: 422,
      message: "The recording is silent.",
    });
  });

  it("explains network failures", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
    const error = await predictAudio(new Blob()).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("network_error");
  });

  it("copes with a non-JSON error body", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("Bad gateway", { status: 502 }));
    await expect(predictAudio(new Blob())).rejects.toMatchObject({
      code: "http_error",
      status: 502,
    });
  });
});
