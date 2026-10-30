// Runs the factoring off the main thread so the page stays responsive.
// The page can terminate this worker to cancel a long factorization.

import init, { factor } from "./pkg/cofox_wasm.js";

const ready = init();

self.onmessage = async (event) => {
  const { id, n } = event.data;
  try {
    await ready;
    const start = performance.now();
    // Composites that couldn't be split come back wrapped in parentheses.
    const factors = factor(n).map((f) =>
      f.startsWith("(")
        ? { value: f.slice(1, -1), prime: false }
        : { value: f, prime: true },
    );
    self.postMessage({ id, factors, ms: performance.now() - start });
  } catch (err) {
    self.postMessage({ id, error: String(err?.message ?? err) });
  }
};
