/**
 * Runs the ONNX model off the main thread.
 *
 * The model is stateful: the worker keeps the LSTM state (h, c) for the
 * drawing so far and feeds each finished stroke once. Messages are handled
 * strictly in order (a promise queue), because inference is async and a
 * reset must never interleave with a stroke.
 */

import * as ort from "onnxruntime-web/wasm"; // WASM-only build: much smaller than the WebGPU one
import { PEN_END, PEN_LIFT, sampleRow } from "./ghost";
import type { FromWorker, ModelMeta, ToWorker } from "./protocol";
import type { Row } from "./preprocess";

const GHOST_TEMPERATURE = 0.4;
const GHOST_MAX_STROKES = 2;
const GHOST_MAX_STEPS = 60;

ort.env.wasm.numThreads = 1; // threads need cross-origin isolation headers

let session: ort.InferenceSession;
let meta: ModelMeta;
let h: Float32Array;
let c: Float32Array;
let started = false;

const post = (msg: FromWorker) => postMessage(msg);

interface StepOut {
  probs: Float32Array;
  mdn: Float32Array;
  h: Float32Array;
  c: Float32Array;
}

async function run(rows: Row[] | number[][], cls: number, h0: Float32Array, c0: Float32Array): Promise<StepOut> {
  const T = rows.length;
  const stateDims = [meta.layers, 1, meta.hidden];
  const out = await session.run({
    x: new ort.Tensor("float32", Float32Array.from(rows.flat()), [1, T, 5]),
    cls: new ort.Tensor("int64", BigInt64Array.from([BigInt(cls)]), [1]),
    h0: new ort.Tensor("float32", h0, stateDims),
    c0: new ort.Tensor("float32", c0, stateDims),
  });
  return {
    probs: out.probs.data as Float32Array,
    mdn: out.mdn.data as Float32Array,
    h: out.h1.data as Float32Array,
    c: out.c1.data as Float32Array,
  };
}

function argmax(a: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}

async function init() {
  const base = import.meta.env.BASE_URL;
  meta = await (await fetch(`${base}model.json`)).json();
  session = await ort.InferenceSession.create(`${base}model.onnx`, { executionProviders: ["wasm"] });
  reset();
  post({ type: "ready", meta, backend: "wasm" });
}

function reset() {
  h = new Float32Array(meta.layers * meta.hidden);
  c = new Float32Array(meta.layers * meta.hidden);
  started = false;
}

async function sampleGhost(cls: number, mdn: Float32Array, h0: Float32Array, c0: Float32Array): Promise<Row[]> {
  const rows: Row[] = [];
  let params = mdn;
  let hs = h0;
  let cs = c0;
  let strokes = 0;
  for (let step = 0; step < GHOST_MAX_STEPS; step++) {
    const s = sampleRow(params, meta.n_mix, GHOST_TEMPERATURE);
    if (s.pen === PEN_END) break;
    const row: Row = [s.dx, s.dy, s.pen === PEN_LIFT ? 0 : 1, s.pen === PEN_LIFT ? 1 : 0, 0];
    rows.push(row);
    if (s.pen === PEN_LIFT && ++strokes >= GHOST_MAX_STROKES) break;
    const out = await run([row], cls, hs, cs);
    params = out.mdn;
    hs = out.h;
    cs = out.c;
  }
  return rows;
}

async function stroke(id: number, rows: Row[], wantGhost: boolean) {
  const t0 = performance.now();
  const C = meta.classes.length;
  const D = 6 * meta.n_mix + 3;
  // The first stroke starts with SOS; its output (the prior) is not reported.
  const input: number[][] = started ? rows : [meta.sos, ...rows];
  const skip = started ? 0 : 1;
  started = true;

  // Feed all but the last row, then the last row on its own: the MDN head
  // depends on the class, and we only know the top class after the last row.
  // Keeping the state before the last row lets us redo it for that class.
  let hPrev = h;
  let cPrev = c;
  const probs: number[][] = [];
  if (input.length > 1) {
    const head = await run(input.slice(0, -1), 0, h, c);
    for (let t = skip; t < input.length - 1; t++) probs.push(Array.from(head.probs.subarray(t * C, (t + 1) * C)));
    hPrev = head.h;
    cPrev = head.c;
  }
  const last = await run(input.slice(-1), 0, hPrev, cPrev);
  probs.push(Array.from(last.probs));
  h = last.h;
  c = last.c;

  let ghost = null;
  if (wantGhost) {
    const top = argmax(last.probs);
    const mdn = top === 0 ? last.mdn : (await run(input.slice(-1), top, hPrev, cPrev)).mdn;
    ghost = { cls: top, rows: await sampleGhost(top, mdn.subarray(0, D), h, c) };
  }
  post({ type: "stroke", id, probs, ghost, ms: performance.now() - t0 });
}

let queue: Promise<unknown> = init().catch((e) => post({ type: "error", message: String(e) }));

onmessage = (e: MessageEvent<ToWorker>) => {
  const msg = e.data;
  queue = queue
    .then(() => (msg.type === "reset" ? reset() : stroke(msg.id, msg.rows, msg.ghost)))
    .catch((err) => post({ type: "error", message: String(err) }));
};
