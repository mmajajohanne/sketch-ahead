import "./style.css";
import { attribution, BeliefHistory, classColor, decide, entropy, renderChart } from "./belief";
import { DrawingCanvas } from "./canvas";
import { rowsToPoints, simplifyStroke, strokeToRows, type Pt, type Row } from "./preprocess";
import type { FromWorker, ModelMeta, ToWorker } from "./protocol";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const status = $("status");
const guess = $("guess");
const decided = $("decided");
const bars = $("bars");
const strokesEl = $("strokes");
const chart = document.getElementById("chart") as unknown as SVGSVGElement;
const ghostToggle = $<HTMLInputElement>("ghost-toggle");

const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
const send = (m: ToWorker) => worker.postMessage(m);

interface StrokeData {
  rows: Row[];
  /** Last point (normalized) after this stroke: where the next offsets start. */
  last: Pt;
}

let meta: ModelMeta | null = null;
let strokes: StrokeData[] = [];
const history = new BeliefHistory();
// Replies are matched to a generation: undo/clear bumps it, stale replies are dropped.
let generation = 0;
let pending = 0;

const canvas = new DrawingCanvas($("canvas") as HTMLCanvasElement, onStroke, () => {});

function onStroke(raw: Pt[]) {
  if (!meta) return;
  const simplified = simplifyStroke(raw);
  const prev = strokes.length ? strokes[strokes.length - 1].last : null;
  const { rows, last } = strokeToRows(simplified, prev);
  strokes.push({ rows, last });
  feed(strokes.length - 1, ghostToggle.checked);
}

function feed(index: number, ghost: boolean) {
  pending++;
  send({ type: "stroke", id: generation * 10000 + index, rows: strokes[index].rows, ghost });
  render();
}

function replay() {
  generation++;
  pending = 0;
  history.clear();
  send({ type: "reset" });
  strokes.forEach((_, i) => feed(i, ghostToggle.checked && i === strokes.length - 1));
  render();
}

worker.onmessage = (e: MessageEvent<FromWorker>) => {
  const msg = e.data;
  if (msg.type === "ready") {
    meta = msg.meta;
    status.textContent = `Model loaded (${meta.classes.length} classes). Draw one of: ${meta.classes.join(", ")}.`;
    render();
  } else if (msg.type === "error") {
    status.textContent = `Error: ${msg.message}`;
  } else if (msg.type === "stroke") {
    if (Math.floor(msg.id / 10000) !== generation) return;
    pending--;
    history.add(msg.probs);
    if (msg.ghost && msg.id % 10000 === strokes.length - 1) {
      const start = strokes[strokes.length - 1].last;
      canvas.setGhost({ start, points: rowsToPoints(msg.ghost.rows, start) });
    }
    const points = strokes.reduce((n, s) => n + s.rows.length, 0);
    status.textContent =
      `${strokes.length} strokes · ${points} points · ${msg.ms.toFixed(0)} ms` +
      (meta && points >= meta.max_len ? " · longer than any training drawing" : "");
    render();
  }
};

$("undo").addEventListener("click", () => {
  if (!strokes.length) return;
  strokes.pop();
  canvas.undo();
  replay();
});

$("clear").addEventListener("click", () => {
  strokes = [];
  canvas.clear();
  replay();
});

ghostToggle.addEventListener("change", () => {
  if (!ghostToggle.checked) canvas.setGhost(null);
  else if (strokes.length) replay();
});

function render() {
  const classes = meta?.classes ?? [];
  const cur = history.current;
  const perStroke = history.perStroke;
  const decision = decide(perStroke);
  renderChart(chart, history, classes, decision);

  if (!cur) {
    guess.textContent = meta ? "Draw something…" : "Loading model…";
    decided.textContent = "";
    bars.replaceChildren();
    strokesEl.replaceChildren();
    return;
  }

  const order = cur.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
  const [pTop, top] = order[0];
  guess.innerHTML = "";
  guess.append(`I think it's ${/^[aeiou]/.test(classes[top]) ? "an" : "a"} `);
  const name = document.createElement("strong");
  name.textContent = classes[top];
  name.style.color = classColor(top);
  guess.append(name, ` · ${(pTop * 100).toFixed(0)}%`);

  decided.textContent = decision
    ? `Decided on ${classes[decision.cls]} after stroke ${decision.stroke + 1}` +
      (pending ? "" : ` · uncertainty ${entropy(cur).toFixed(2)} bits`)
    : `Not sure yet · uncertainty ${entropy(cur).toFixed(2)} bits`;

  bars.replaceChildren(
    ...order.map(([p, i]) => {
      const row = document.createElement("div");
      row.className = "bar";
      row.innerHTML = `<span class="bar-name"></span><span class="bar-track"><span class="bar-fill"></span></span><span class="bar-val"></span>`;
      row.querySelector<HTMLElement>(".bar-name")!.textContent = classes[i];
      const fill = row.querySelector<HTMLElement>(".bar-fill")!;
      fill.style.width = `${(p * 100).toFixed(1)}%`;
      fill.style.background = classColor(i);
      row.querySelector<HTMLElement>(".bar-val")!.textContent = `${(p * 100).toFixed(0)}%`;
      return row;
    }),
  );

  // Stroke attribution: how much each stroke moved the current top guess.
  const attr = attribution(perStroke, top);
  const maxAbs = Math.max(0.05, ...attr.map(Math.abs));
  strokesEl.replaceChildren(
    ...attr.map((d, k) => {
      const b = document.createElement("button");
      b.className = "attr";
      b.title = `Stroke ${k + 1}: ${d >= 0 ? "+" : ""}${(d * 100).toFixed(0)} points for ${classes[top]}`;
      const h = Math.max(2, (Math.abs(d) / maxAbs) * 28);
      b.innerHTML = `<span class="attr-bar ${d >= 0 ? "up" : "down"}" style="height:${h}px"></span><span class="attr-label">${k + 1}</span>`;
      b.addEventListener("mouseenter", () => canvas.setHighlight(k));
      b.addEventListener("mouseleave", () => canvas.setHighlight(-1));
      b.addEventListener("focus", () => canvas.setHighlight(k));
      b.addEventListener("blur", () => canvas.setHighlight(-1));
      return b;
    }),
  );
}

render();
