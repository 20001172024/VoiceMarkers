import { useEffect, useState } from "react";

export type Route = "analyse" | "about";

/** Map a URL hash to a page. Unknown hashes fall back to the main page. */
export function routeFromHash(hash: string): Route {
  return hash.replace(/^#\/?/, "") === "about" ? "about" : "analyse";
}

/** Tiny hash router: two pages don't need a routing library. */
export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => routeFromHash(window.location.hash));
  useEffect(() => {
    const onChange = () => {
      setRoute(routeFromHash(window.location.hash));
      window.scrollTo?.(0, 0);
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}
