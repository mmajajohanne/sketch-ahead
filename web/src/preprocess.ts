/**
 * Live strokes -> stroke-5 rows, exactly as train/data/strokes.py does it.
 *
 * IMPORTANT: `strokeToRows` must match `drawing_to_points` + `normalize` +
 * `to_stroke5` in Python. test/preprocess.test.ts checks this against a
 * fixture that train/tests/test_web_fixture.py checks against Python.
 *
 * The training data is Quick, Draw! "simplified": coordinates in 0-255,
 * strokes resampled and simplified with Ramer-Douglas-Peucker (epsilon 2),
 * rounded to integers. Live strokes go through the same RDP + rounding
 * (`simplifyStroke`) so the model sees offsets of the same size as in training.
 */

export const CANVAS = 256;
export const ORIGIN: Pt = [0.5, 0.5];
export const OFFSET_SCALE = 0.11;
export const SOS: Row = [0, 0, 1, 0, 0];
export const RDP_EPSILON = 2;

export type Pt = [number, number];
/** (dx, dy, p_down, p_lift, p_end), offsets divided by OFFSET_SCALE. */
export type Row = [number, number, number, number, number];

/** Ramer-Douglas-Peucker simplification (keeps first and last point). */
export function rdp(points: Pt[], epsilon = RDP_EPSILON): Pt[] {
  if (points.length < 3) return points.slice();
  const keep = new Uint8Array(points.length);
  keep[0] = keep[points.length - 1] = 1;
  const stack: [number, number][] = [[0, points.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop()!;
    const [ax, ay] = points[a];
    const [bx, by] = points[b];
    const dx = bx - ax;
    const dy = by - ay;
    const len = Math.hypot(dx, dy);
    let best = -1;
    let bestDist = epsilon;
    for (let i = a + 1; i < b; i++) {
      const [px, py] = points[i];
      const d = len === 0 ? Math.hypot(px - ax, py - ay) : Math.abs(dy * px - dx * py + bx * ay - by * ax) / len;
      if (d > bestDist) {
        bestDist = d;
        best = i;
      }
    }
    if (best >= 0) {
      keep[best] = 1;
      stack.push([a, best], [best, b]);
    }
  }
  return points.filter((_, i) => keep[i]);
}

/** Raw pointer points in canvas units (0-255) -> simplified integer points. */
export function simplifyStroke(raw: Pt[]): Pt[] {
  const clamped = raw.map(([x, y]) => [clamp(x, 0, 255), clamp(y, 0, 255)] as Pt);
  const out: Pt[] = [];
  for (const [x, y] of rdp(clamped)) {
    const p: Pt = [Math.round(x), Math.round(y)];
    const last = out[out.length - 1];
    if (!last || last[0] !== p[0] || last[1] !== p[1]) out.push(p);
  }
  return out;
}

/**
 * One stroke (canvas units) -> stroke-5 rows. `prev` is the previous point in
 * normalized units (null for the first stroke: offsets start at ORIGIN).
 * Returns the rows and the stroke's last point (normalized) for the next call.
 */
export function strokeToRows(stroke: Pt[], prev: Pt | null): { rows: Row[]; last: Pt } {
  let [px, py] = prev ?? ORIGIN;
  const rows: Row[] = stroke.map(([x, y], i) => {
    const nx = x / CANVAS;
    const ny = y / CANVAS;
    const lift = i === stroke.length - 1 ? 1 : 0;
    const row: Row = [(nx - px) / OFFSET_SCALE, (ny - py) / OFFSET_SCALE, 1 - lift, lift, 0];
    px = nx;
    py = ny;
    return row;
  });
  return { rows, last: [px, py] };
}

/** Whole drawing in ndjson layout ([[xs, ys], ...]) -> rows (no SOS, no end). */
export function drawingToRows(drawing: [number[], number[]][]): Row[] {
  const out: Row[] = [];
  let prev: Pt | null = null;
  for (const [xs, ys] of drawing) {
    const { rows, last } = strokeToRows(xs.map((x, i) => [x, ys[i]] as Pt), prev);
    out.push(...rows);
    prev = last;
  }
  return out;
}

/** Inverse for rendering: rows -> absolute normalized points, starting at `start`. */
export function rowsToPoints(rows: Row[], start: Pt): { x: number; y: number; lift: boolean }[] {
  let [x, y] = start;
  return rows.map((r) => {
    x += r[0] * OFFSET_SCALE;
    y += r[1] * OFFSET_SCALE;
    return { x, y, lift: r[3] > 0.5 };
  });
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}
