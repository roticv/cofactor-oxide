// Runs the factoring off the main thread so the page stays responsive. The
// page starts several of these and terminates them to cancel. Each message
// has a type:
//
// - "factor": the pipeline up to ECM, with this worker's share of the
//   curves. Composites that ECM leaves come back unsplit.
// - "sieve": the quadratic sieve on one number, forever, posting relations
//   as they're found, until the page terminates the worker.
// - "collect" and then "relations": gathers the relations that the page
//   forwards from the sieving workers, and finds the factor once there are
//   enough. This runs in a worker of its own.

import init, { Collector, Sieve, factor } from "./pkg/cofox_wasm.js";

const ready = init();

// Progress is sent at most this often, plus whenever the stage changes.
const PROGRESS_MS = 100;
// Each batch of relations takes about this long to sieve.
const BATCH_MS = 100;

let collector = null;
let solving = false;
let sent = 0;

self.onmessage = async (event) => {
  const message = event.data;
  const { id } = message;
  try {
    await ready;
    switch (message.type) {
      case "factor":
        return runFactor(message);
      case "sieve":
        return runSieve(message);
      case "collect":
        collector = new Collector(message.n);
        if (!collector.needs_sieving()) {
          return self.postMessage({ id, immediate: collector.immediate() ?? null });
        }
        return self.postMessage({ id, ready: true });
      case "relations":
        return addRelations(message);
    }
  } catch (err) {
    self.postMessage({ id, error: String(err?.message ?? err) });
  }
};

function runFactor({ id, n, ecmDigits, worker, workers }) {
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
}

function runSieve({ id, n, seed }) {
  const sieve = new Sieve(n, seed);
  // Start with one polynomial per batch and adjust to the batch time.
  let polys = 1;
  for (;;) {
    const start = performance.now();
    const relations = sieve.sieve(polys);
    const elapsed = performance.now() - start;
    if (elapsed < BATCH_MS / 2) polys *= 2;
    else if (elapsed > BATCH_MS * 2 && polys > 1) polys = Math.floor(polys / 2);
    self.postMessage({ id, relations }, [relations.buffer]);
  }
}

function addRelations({ id, relations }) {
  if (solving) return;
  collector.add(relations);
  const done = collector.relations();
  const total = collector.target();
  const now = performance.now();
  if (now - sent >= PROGRESS_MS || collector.is_complete()) {
    sent = now;
    self.postMessage({ id, progress: { method: "qs", bits: collector.bits(), done, total } });
  }
  if (collector.is_complete()) {
    solving = true;
    self.postMessage({ id, solving: true });
    self.postMessage({ id, found: collector.finish() ?? null });
  }
}
