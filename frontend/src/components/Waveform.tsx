import { useEffect, useRef } from "react";

/** Live oscilloscope of the microphone signal. Decorative: hidden from screen readers. */
export function Waveform({ analyser }: { analyser: AnalyserNode | null }) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const el = canvas.current;
    const ctx = el?.getContext?.("2d");
    if (!el || !ctx || !analyser) return;
    const data = new Uint8Array(analyser.fftSize);
    const ink = getComputedStyle(el).getPropertyValue("color") || "#191919";
    let frame = 0;

    const draw = () => {
      analyser.getByteTimeDomainData(data);
      const { width, height } = el;
      ctx.clearRect(0, 0, width, height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = ink;
      ctx.beginPath();
      data.forEach((v, i) => {
        const x = (i / (data.length - 1)) * width;
        const y = (v / 255) * height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      frame = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(frame);
  }, [analyser]);

  return (
    <canvas
      ref={canvas}
      width={600}
      height={80}
      aria-hidden="true"
      className="h-20 w-full rounded-md bg-surface-2 text-accent"
    />
  );
}
