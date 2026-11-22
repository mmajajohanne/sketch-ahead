/**
 * Belief over classes as the drawing grows: decision with hysteresis, stroke
 * attribution, and the chart.
 *
 * Decision (hysteresis): the model "decides" on a class when its probability
 * after a stroke reaches ENTER, and only lets go when that probability falls
 * below EXIT. Two thresholds instead of one stop the marker from flickering
 * when a probability hovers around a single cut-off.
 *
 * Attribution: how much each stroke moved the probability of the current top
 * class (Δp per stroke). The first stroke is measured against a uniform prior.
 */

export const ENTER = 0.7;
export const EXIT = 0.4;

export interface Decision {
  cls: number;
  /** 0-based index of the stroke after which the decision was made. */
  stroke: number;
}

export function argmax(a: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}

/** Decision after the last stroke, replaying the hysteresis over all strokes. */
export function decide(perStroke: number[][], enter = ENTER, exit = EXIT): Decision | null {
  let d: Decision | null = null;
  perStroke.forEach((p, k) => {
    const top = argmax(p);
    if (d && p[d.cls] < exit) d = null;
    if (!d && p[top] >= enter) d = { cls: top, stroke: k };
  });
  return d;
}

/** Δp of class `cls` for each stroke. */
export function attribution(perStroke: number[][], cls: number): number[] {
  if (!perStroke.length) return [];
  const prior = 1 / perStroke[0].length;
  return perStroke.map((p, k) => p[cls] - (k === 0 ? prior : perStroke[k - 1][cls]));
}

/** Entropy in bits. */
export function entropy(p: ArrayLike<number>): number {
  let h = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > 0) h -= p[i] * Math.log2(p[i]);
  return h;
}

export class BeliefHistory {
  /** Probabilities after every point. */
  points: number[][] = [];
  /** Index into `points` of the last point of each stroke. */
  strokeEnds: number[] = [];

  add(probs: number[][]) {
    this.points.push(...probs);
    this.strokeEnds.push(this.points.length - 1);
  }

  clear() {
    this.points = [];
    this.strokeEnds = [];
  }

  get perStroke(): number[][] {
    return this.strokeEnds.map((i) => this.points[i]);
  }

  get current(): number[] | null {
    return this.points.length ? this.points[this.points.length - 1] : null;
  }
}

// ---------------------------------------------------------------- rendering

const SVG = "http://www.w3.org/2000/svg";
const CHART_LINES = 4;

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, parent?: Element) {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  parent?.appendChild(e);
  return e;
}

export function classColor(i: number): string {
  return `var(--c${i % 10})`;
}

/** Line chart: probability of the top classes after every point. */
export function renderChart(svg: SVGSVGElement, hist: BeliefHistory, classes: string[], decision: Decision | null) {
  svg.replaceChildren();
  const W = 520;
  const H = 200;
  const pad = { l: 34, r: 74, t: 10, b: 24 };
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const n = hist.points.length;
  const x = (i: number) => pad.l + (n <= 1 ? 0 : (i / (n - 1)) * (W - pad.l - pad.r));
  const y = (p: number) => pad.t + (1 - p) * (H - pad.t - pad.b);

  for (const p of [0, 0.5, 1]) {
    el("line", { x1: pad.l, x2: W - pad.r, y1: y(p), y2: y(p), class: "grid" }, svg);
    el("text", { x: pad.l - 6, y: y(p) + 4, class: "axis", "text-anchor": "end" }, svg).textContent = `${p * 100}%`;
  }
  if (!n) {
    el("text", { x: (W - pad.r + pad.l) / 2, y: H / 2, class: "axis", "text-anchor": "middle" }, svg).textContent =
      "Start drawing";
    return;
  }

  hist.strokeEnds.forEach((i, k) => {
    el("line", { x1: x(i), x2: x(i), y1: pad.t, y2: H - pad.b, class: "stroke-end" }, svg);
    if (hist.strokeEnds.length <= 20 || k % 2 === 0)
      el("text", { x: x(i), y: H - 8, class: "axis", "text-anchor": "middle" }, svg).textContent = String(k + 1);
  });

  if (decision) {
    const xi = x(hist.strokeEnds[decision.stroke]);
    el("line", { x1: xi, x2: xi, y1: pad.t, y2: H - pad.b, class: "decided" }, svg);
  }

  const cur = hist.current!;
  const order = cur.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, CHART_LINES);
  // Labels at the line ends, pushed apart so they never overlap.
  const labelY: number[] = [];
  order.forEach(([p], k) => labelY.push(k ? Math.max(y(p), labelY[k - 1] + 13) : y(p)));
  const overflow = labelY[labelY.length - 1] - (H - pad.b);
  if (overflow > 0) for (let k = 0; k < labelY.length; k++) labelY[k] -= overflow;
  order.forEach(([, cls], k) => {
    const d = hist.points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p[cls]).toFixed(1)}`).join("");
    el("path", { d, class: "belief-line", stroke: classColor(cls) }, svg);
    el("text", { x: W - pad.r + 6, y: labelY[k] + 4, class: "line-label", fill: classColor(cls) }, svg).textContent =
      classes[cls];
  });
}
