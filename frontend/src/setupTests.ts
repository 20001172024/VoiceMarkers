import "@testing-library/jest-dom/vitest";

// jsdom has no layout engine: give Recharts' ResponsiveContainer a ResizeObserver.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= ResizeObserverStub as unknown as typeof ResizeObserver;
