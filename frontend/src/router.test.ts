import { routeFromHash } from "./router";

describe("routeFromHash", () => {
  it.each([
    ["", "analyse"],
    ["#/", "analyse"],
    ["#/about", "about"],
    ["#about", "about"],
    ["#/nonsense", "analyse"],
  ])("%s -> %s", (hash, route) => {
    expect(routeFromHash(hash)).toBe(route);
  });
});
