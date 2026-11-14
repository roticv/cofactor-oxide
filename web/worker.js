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
// - "gnfs-collect", "gnfs-sieve" and "gnfs-relations": the same for the
//   number field sieve. The collector chooses the polynomials, which the
//   page passes to the sieving workers, and once there are enough
//   relations it filters the matrix and splits it into parts, one for each
//   worker.
// - "matrix-part": one part of that matrix, solved by block Lanczos in step
//   with the other workers' parts, summing vectors with them directly over
//   the message ports the page gives each pair of workers. Each worker
//   sends back its part's words of the dependencies.
// - "gnfs-matrix-done": the parts' words, joined, for the collector, which
//   tries the dependencies one at a time.

// Browsers with the wide-arithmetic proposal get a build that uses its
// 64×64→128-bit multiply, i64.mul_wide_u, which this tiny module tests for:
// (func (param i64 i64) (result i64 i64) local.get 0 local.get 1 i64.mul_wide_u)
// That build also uses SIMD, which every such browser has. It is optional
// locally, so a missing one falls back to the other.
const WIDE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 8, 1, 96, 2, 126, 126, 2, 126, 126, 3, 2, 1, 0,
  10, 10, 1, 8, 0, 32, 0, 32, 1, 252, 22, 11,
]);
let Collector, GnfsCollector, GnfsSieve, MatrixPart, Sieve, factor;
const ready = (
  WebAssembly.validate(WIDE)
    ? import("./pkg-wide/cofox_wasm.js").catch(() => import("./pkg/cofox_wasm.js"))
    : import("./pkg/cofox_wasm.js")
).then((pkg) => {
  ({ Collector, GnfsCollector, GnfsSieve, MatrixPart, Sieve, factor } = pkg);
  return pkg.default();
});

// Progress is sent at most this often, plus whenever the stage changes.
const PROGRESS_MS = 100;
// Each batch of relations takes about this long to sieve.
const BATCH_MS = 100;

let collector = null;
let solving = false;
// The workers that solve the matrix together, one part each.
let matrixWorkers = 1;
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
      case "gnfs-collect":
        collector = new GnfsCollector(message.n);
        matrixWorkers = message.workers ?? 1;
        return self.postMessage({ id, poly: collector.poly() });
      case "gnfs-sieve":
        return runGnfsSieve(message);
      case "gnfs-relations":
        return addGnfsRelations(message);
      case "matrix-part":
        return runMatrixPart(message);
      case "gnfs-matrix-done":
        return tryDependencies(id, message.parts);
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

function runGnfsSieve({ id, n, poly, worker, workers }) {
  const sieve = new GnfsSieve(n, poly, worker, workers);
  for (;;) {
    const relations = sieve.sieve();
    if (relations) self.postMessage({ id, gnfsRelations: relations });
  }
}

function addGnfsRelations({ id, relations }) {
  if (solving) return;
  collector.add(relations);
  const bits = collector.bits();
  const ready = collector.is_ready();
  const now = performance.now();
  if (now - sent >= PROGRESS_MS || ready) {
    sent = now;
    const done = collector.relations();
    self.postMessage({ id, progress: { method: "gnfs", bits, done, total: collector.ideals() } });
  }
  if (!ready) return;
  solving = true;
  self.postMessage({ id, solving: true });
  const count = collector.split_matrix(matrixWorkers);
  if (count === 0) return tryDependencies(id, []);
  const parts = [];
  for (let i = 0; i < count; i++) parts.push(collector.matrix_part(i));
  self.postMessage({ id, matrixParts: parts }, parts.map((p) => p.buffer));
}

// Takes the matrix parts' words of the dependencies and tries each
// dependency's square roots.
function tryDependencies(id, parts) {
  const bits = collector.bits();
  const words = new BigUint64Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    words.set(p, at);
    at += p.length;
  }
  const dependencies = collector.finish_matrix(words);
  for (let i = 0; i < dependencies; i++) {
    self.postMessage({ id, progress: { method: "gnfs-sqrt", bits, done: i, total: dependencies } });
    const found = collector.try_dependency(i);
    if (found !== undefined) return self.postMessage({ id, found });
  }
  self.postMessage({ id, found: null });
}

// Solves one part of the matrix with the other workers' parts: each step
// gives a vector to sum with theirs, and goes on from the sum.
async function runMatrixPart({ id, data, index, ports }) {
  const part = new MatrixPart(data);
  const inboxes = ports.map((port) => port && inbox(port));
  let vector = part.start();
  for (;;) {
    const sum = await sumWithPeers(vector, index, ports, inboxes);
    const next = part.step(sum);
    if (next === undefined) break;
    vector = next;
    const now = performance.now();
    if (index === 0 && now - sent >= PROGRESS_MS) {
      sent = now;
      self.postMessage({ id, progress: { method: "gnfs-matrix", done: part.steps(), total: part.expected_steps(), workers: ports.length } });
    }
  }
  const words = part.dependencies();
  part.free();
  self.postMessage({ id, matrixDeps: words, index }, [words.buffer]);
}

// The messages from a port, in order, as promises.
function inbox(port) {
  const queue = [];
  const waiting = [];
  port.onmessage = (event) => (waiting.length > 0 ? waiting.shift()(event.data) : queue.push(event.data));
  return () => (queue.length > 0 ? Promise.resolve(queue.shift()) : new Promise((resolve) => waiting.push(resolve)));
}

// The xor of `vector` with every peer's, by reduce-scatter then all-gather:
// worker j sums the jth of k slices from everyone and sends the sum back to
// all, so each sends and receives about two vectors' worth whatever k is.
// A port's messages arrive in order, and a peer can't send its next slice
// before it has this step's sums, so the next message from each peer is
// always the one wanted.
async function sumWithPeers(vector, me, ports, inboxes) {
  const k = ports.length;
  if (k === 1) return vector;
  const bound = (j) => Math.floor((j * vector.length) / k);
  for (let j = 0; j < k; j++) {
    if (j === me) continue;
    const slice = vector.slice(bound(j), bound(j + 1));
    ports[j].postMessage(slice, [slice.buffer]);
  }
  const mine = vector.subarray(bound(me), bound(me + 1));
  for (const slice of await Promise.all(inboxes.map((next, j) => (j === me ? null : next())))) {
    if (slice) for (let i = 0; i < slice.length; i++) mine[i] ^= slice[i];
  }
  for (let j = 0; j < k; j++) {
    if (j === me) continue;
    const copy = mine.slice();
    ports[j].postMessage(copy, [copy.buffer]);
  }
  const sums = await Promise.all(inboxes.map((next, j) => (j === me ? null : next())));
  sums.forEach((slice, j) => slice && vector.set(slice, bound(j)));
  return vector;
}
