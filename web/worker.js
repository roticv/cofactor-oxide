// Runs the factoring off the main thread so the page stays responsive. The
// page starts several of these, each taking its share of the ECM curves,
// and terminates them to cancel.

import init, { factor } from "./pkg/cofox_wasm.js";

const ready = init();

// Progress is sent at most this often, plus whenever the stage changes.
const PROGRESS_MS = 100;

self.onmessage = async (event) => {
  const { id, n, ecmDigits, worker, workers } = event.data;
  try {
    await ready;
    let sent = 0;
    let lastStage = "";
    const progress = (method, bits, digits, done, total) => {
      const stage = `${method} ${bits} ${digits}`;
      const now = performance.now();
      if (stage === lastStage && now - sent < PROGRESS_MS) return;
      lastStage = stage;
      sent = now;
      self.postMessage({ id, progress: { method, bits, digits, done, total } });
    };
    // Composites that couldn't be split come back wrapped in parentheses.
    const factors = factor(n, ecmDigits, worker, workers, progress).map((f) =>
      f.startsWith("(")
        ? { value: f.slice(1, -1), prime: false }
        : { value: f, prime: true },
    );
    self.postMessage({ id, factors });
  } catch (err) {
    self.postMessage({ id, error: String(err?.message ?? err) });
  }
};
