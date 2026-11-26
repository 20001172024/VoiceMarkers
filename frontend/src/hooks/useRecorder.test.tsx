import { act, renderHook } from "@testing-library/react";
import { microphoneErrorMessage, useRecorder } from "./useRecorder";

/** Minimal MediaRecorder stand-in: emits one chunk when stopped. */
class FakeMediaRecorder {
  static isTypeSupported = (type: string) => type === "audio/webm;codecs=opus";
  state: "inactive" | "recording" = "inactive";
  mimeType: string;
  ondataavailable: ((e: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  constructor(_stream: MediaStream, options?: { mimeType?: string }) {
    this.mimeType = options?.mimeType ?? "";
  }
  start() {
    this.state = "recording";
  }
  stop() {
    this.state = "inactive";
    this.ondataavailable?.({ data: new Blob(["audio"], { type: this.mimeType }) });
    this.onstop?.();
  }
}

const track = { stop: vi.fn() };
const stream = { getTracks: () => [track] } as unknown as MediaStream;

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("MediaRecorder", FakeMediaRecorder);
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia: vi.fn().mockResolvedValue(stream) },
  });
  URL.createObjectURL = vi.fn(() => "blob:recording");
  URL.revokeObjectURL = vi.fn();
  track.stop.mockClear();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("useRecorder", () => {
  it("counts down, records for a fixed time, then releases the microphone", async () => {
    const { result } = renderHook(() => useRecorder({ countdownSeconds: 2, recordSeconds: 3 }));

    await act(async () => {
      await result.current.start();
    });
    expect(result.current.state).toEqual({ status: "countdown", remaining: 2 });

    act(() => vi.advanceTimersByTime(1000));
    expect(result.current.state).toEqual({ status: "countdown", remaining: 1 });

    act(() => vi.advanceTimersByTime(1000));
    expect(result.current.state.status).toBe("recording");

    act(() => vi.advanceTimersByTime(3100));
    const state = result.current.state;
    expect(state.status).toBe("recorded");
    if (state.status === "recorded") {
      expect(state.blob.type).toBe("audio/webm;codecs=opus");
      expect(state.url).toBe("blob:recording");
    }
    expect(track.stop).toHaveBeenCalled(); // microphone released
  });

  it("can be stopped early", async () => {
    const { result } = renderHook(() => useRecorder({ countdownSeconds: 0, recordSeconds: 5 }));
    await act(async () => {
      await result.current.start();
    });
    expect(result.current.state.status).toBe("recording");
    act(() => result.current.stop());
    expect(result.current.state.status).toBe("recorded");
  });

  it("explains a blocked microphone", async () => {
    const denied = new DOMException("denied", "NotAllowedError");
    (navigator.mediaDevices.getUserMedia as ReturnType<typeof vi.fn>).mockRejectedValue(denied);
    const { result } = renderHook(() => useRecorder());
    await act(async () => {
      await result.current.start();
    });
    expect(result.current.state).toEqual({
      status: "error",
      message: expect.stringMatching(/blocked/),
    });
  });

  it("reports when the browser cannot record", async () => {
    vi.stubGlobal("MediaRecorder", undefined);
    const { result } = renderHook(() => useRecorder());
    await act(async () => {
      await result.current.start();
    });
    expect(result.current.state.status).toBe("error");
  });

  it("reset returns to idle and frees the recording", async () => {
    const { result } = renderHook(() => useRecorder({ countdownSeconds: 0, recordSeconds: 1 }));
    await act(async () => {
      await result.current.start();
    });
    act(() => vi.advanceTimersByTime(1100));
    act(() => result.current.reset());
    expect(result.current.state).toEqual({ status: "idle" });
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:recording");
  });

  it("releases the microphone if the component unmounts mid-recording", async () => {
    const { result, unmount } = renderHook(() => useRecorder({ countdownSeconds: 0 }));
    await act(async () => {
      await result.current.start();
    });
    unmount();
    expect(track.stop).toHaveBeenCalled();
  });
});

describe("microphoneErrorMessage", () => {
  it.each([
    ["NotFoundError", /No microphone/],
    ["NotReadableError", /another app/],
    ["Whatever", /not available/],
  ])("%s", (name, message) => {
    expect(microphoneErrorMessage(new DOMException("x", name))).toMatch(message);
  });
});
