/**
 * Runs the two ONNX graphs off the main thread.
 *
 * classifier.onnx is stateful: the worker keeps its LSTM state for the
 * drawing so far and feeds each finished stroke once.
 *
 * generator.onnx is conditioned on the class at every step, so its state
 * depends on which class we ask for. For a ghost it is rerun over the whole
 * drawing for the current top class (one batched call), then stepped one
 * sampled point at a time.
 *
 * Messages are handled strictly in order (a promise queue), because inference
 * is async and a reset must never interleave with a stroke.
 */

import * as ort from "onnxruntime-web/wasm"; // WASM-only build: much smaller than the WebGPU one
import { PEN_END, PEN_LIFT, sampleRow } from "./ghost";
import type { FromWorker, ModelMeta, ToWorker } from "./protocol";
import type { Pt, Row } from "./preprocess";

// Ghost settings. The ghost is a suggestion, so it is only shown when the
// model is fairly sure of the class, is kept to one stroke (errors compound
// with every sampled point), and is redrawn if it leaves the canvas.
const GHOST_TEMPERATURE = 0.25;
const GHOST_MIN_CONFIDENCE = 0.6;
const GHOST_MAX_STROKES = 1;
const GHOST_MAX_STEPS = 40;
const GHOST_TRIES = 4;
const GHOST_MARGIN = 0.02; // how far outside the canvas (normalized) a ghost may go

ort.env.wasm.numThreads = 1; // threads need cross-origin isolation headers

let classifier: ort.InferenceSession;
let generator: ort.InferenceSession;
let meta: ModelMeta;
let h: Float32Array;
let c: Float32Array;
/** Every row fed so far, SOS first: the generator replays this. */
let drawing: number[][] = [];

const post = (msg: FromWorker) => postMessage(msg);

interface State {
  h: Float32Array;
  c: Float32Array;
}

const zeros = () => new Float32Array(meta.layers * meta.hidden);

function inputs(rows: number[][], s: State) {
  const stateDims = [meta.layers, 1, meta.hidden];
  return {
    x: new ort.Tensor("float32", Float32Array.from(rows.flat()), [1, rows.length, 5]),
    h0: new ort.Tensor("float32", s.h, stateDims),
    c0: new ort.Tensor("float32", s.c, stateDims),
  };
}

async function classify(rows: number[][], s: State) {
  const out = await classifier.run(inputs(rows, s));
  return { probs: out.probs.data as Float32Array, h: out.h1.data as Float32Array, c: out.c1.data as Float32Array };
}

async function generate(rows: number[][], cls: number, s: State) {
  const out = await generator.run({
    ...inputs(rows, s),
    cls: new ort.Tensor("int64", BigInt64Array.from([BigInt(cls)]), [1]),
  });
  const mdn = out.mdn.data as Float32Array;
  const D = 6 * meta.n_mix + 3;
  return { mdn: mdn.subarray(mdn.length - D), h: out.h1.data as Float32Array, c: out.c1.data as Float32Array };
}

function argmax(a: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}

async function init() {
  const base = import.meta.env.BASE_URL;
  meta = await (await fetch(`${base}model.json`)).json();
  const opts = { executionProviders: ["wasm"] };
  [classifier, generator] = await Promise.all([
    ort.InferenceSession.create(`${base}classifier.onnx`, opts),
    ort.InferenceSession.create(`${base}generator.onnx`, opts),
  ]);
  reset();
  post({ type: "ready", meta, backend: "wasm" });
}

function reset() {
  h = zeros();
  c = zeros();
  drawing = [];
}

/** One sampled continuation, or null if it leaves the canvas. */
async function sampleOnce(cls: number, mdn: Float32Array, s0: State, start: Pt): Promise<Row[] | null> {
  const rows: Row[] = [];
  let params = mdn;
  let s = s0;
  let strokes = 0;
  let [x, y] = start;
  for (let step = 0; step < GHOST_MAX_STEPS; step++) {
    const p = sampleRow(params, meta.n_mix, GHOST_TEMPERATURE);
    if (p.pen === PEN_END) break;
    x += p.dx * meta.offset_scale;
    y += p.dy * meta.offset_scale;
    if (x < -GHOST_MARGIN || x > 1 + GHOST_MARGIN || y < -GHOST_MARGIN || y > 1 + GHOST_MARGIN) return null;
    const row: Row = [p.dx, p.dy, p.pen === PEN_LIFT ? 0 : 1, p.pen === PEN_LIFT ? 1 : 0, 0];
    rows.push(row);
    if (p.pen === PEN_LIFT && ++strokes >= GHOST_MAX_STROKES) break;
    const out = await generate([row], cls, s);
    params = out.mdn;
    s = out;
  }
  return rows;
}

async function sampleGhost(cls: number, start: Pt): Promise<Row[]> {
  const prefix = await generate(drawing, cls, { h: zeros(), c: zeros() });
  for (let i = 0; i < GHOST_TRIES; i++) {
    const rows = await sampleOnce(cls, prefix.mdn, prefix, start);
    if (rows) return rows;
  }
  return [];
}

async function stroke(id: number, rows: Row[], wantGhost: boolean, start: Pt) {
  const t0 = performance.now();
  const C = meta.classes.length;
  // The first stroke starts with SOS; its output (the prior) is not reported.
  const first = drawing.length === 0;
  const input: number[][] = first ? [meta.sos, ...rows] : rows;
  drawing.push(...input);

  const out = await classify(input, { h, c });
  h = out.h;
  c = out.c;
  const probs: number[][] = [];
  for (let t = first ? 1 : 0; t < input.length; t++) probs.push(Array.from(out.probs.subarray(t * C, (t + 1) * C)));

  const last = probs[probs.length - 1];
  const top = argmax(last);
  const ghost = wantGhost && last[top] >= GHOST_MIN_CONFIDENCE ? { cls: top, rows: await sampleGhost(top, start) } : null;
  post({ type: "stroke", id, probs, ghost, ms: performance.now() - t0 });
}

let queue: Promise<unknown> = init().catch((e) => post({ type: "error", message: String(e) }));

onmessage = (e: MessageEvent<ToWorker>) => {
  const msg = e.data;
  queue = queue
    .then(() => (msg.type === "reset" ? reset() : stroke(msg.id, msg.rows, msg.ghost, msg.start)))
    .catch((err) => post({ type: "error", message: String(err) }));
};
