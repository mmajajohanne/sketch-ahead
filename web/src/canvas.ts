/**
 * Drawing surface. Pointer input is mapped to canvas units 0-255, the same
 * range as Quick, Draw! simplified data. Raw points are kept for display;
 * the caller simplifies them for the model.
 */

import { CANVAS, type Pt } from "./preprocess";

export interface Ghost {
  /** Absolute points in normalized units (0-1), with stroke ends. */
  points: { x: number; y: number; lift: boolean }[];
  /** Where the ghost starts (the drawing's last point, normalized). */
  start: Pt;
}

export class DrawingCanvas {
  strokes: Pt[][] = [];
  private current: Pt[] | null = null;
  private ghost: Ghost | null = null;
  private highlight = -1;
  private ctx: CanvasRenderingContext2D;

  constructor(
    private el: HTMLCanvasElement,
    private onStroke: (raw: Pt[]) => void,
    private onStrokeStart: () => void,
  ) {
    this.ctx = el.getContext("2d")!;
    el.addEventListener("pointerdown", (e) => this.down(e));
    el.addEventListener("pointermove", (e) => this.move(e));
    el.addEventListener("pointerup", (e) => this.up(e));
    el.addEventListener("pointercancel", (e) => this.up(e));
    new ResizeObserver(() => this.resize()).observe(el);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.redraw());
  }

  setGhost(g: Ghost | null) {
    this.ghost = g;
    this.redraw();
  }

  setHighlight(stroke: number) {
    this.highlight = stroke;
    this.redraw();
  }

  undo() {
    this.strokes.pop();
    this.ghost = null;
    this.redraw();
  }

  clear() {
    this.strokes = [];
    this.ghost = null;
    this.redraw();
  }

  private toCanvas(e: PointerEvent): Pt {
    const r = this.el.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * CANVAS, ((e.clientY - r.top) / r.height) * CANVAS];
  }

  private down(e: PointerEvent) {
    if (e.button !== 0) return;
    this.el.setPointerCapture(e.pointerId);
    this.current = [this.toCanvas(e)];
    this.ghost = null;
    this.onStrokeStart();
    this.redraw();
  }

  private move(e: PointerEvent) {
    if (!this.current) return;
    const events = e.getCoalescedEvents?.() ?? [e];
    for (const ev of events.length ? events : [e]) this.current.push(this.toCanvas(ev));
    this.redraw();
  }

  private up(e: PointerEvent) {
    if (!this.current) return;
    if (e.type === "pointerup") this.current.push(this.toCanvas(e));
    const stroke = this.current;
    this.current = null;
    this.strokes.push(stroke);
    this.redraw();
    this.onStroke(stroke);
  }

  private resize() {
    const dpr = window.devicePixelRatio || 1;
    const r = this.el.getBoundingClientRect();
    this.el.width = Math.round(r.width * dpr);
    this.el.height = Math.round(r.height * dpr);
    this.redraw();
  }

  redraw() {
    const { ctx, el } = this;
    const css = getComputedStyle(el);
    const s = el.width / CANVAS; // canvas units -> device pixels
    ctx.clearRect(0, 0, el.width, el.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    if (this.ghost) {
      const g = this.ghost;
      ctx.save();
      ctx.strokeStyle = css.getPropertyValue("--ghost");
      ctx.lineWidth = 2.2 * s;
      ctx.setLineDash([3 * s, 3 * s]);
      ctx.beginPath();
      // The pen is up between the drawing's last point and the ghost's first.
      let penDown = false;
      for (const p of g.points) {
        const px = p.x * el.width;
        const py = p.y * el.height;
        if (penDown) ctx.lineTo(px, py);
        else ctx.moveTo(px, py);
        penDown = !p.lift;
      }
      ctx.stroke();
      ctx.restore();
    }

    const ink = css.getPropertyValue("--ink");
    const accent = css.getPropertyValue("--accent");
    const all = this.current ? [...this.strokes, this.current] : this.strokes;
    all.forEach((stroke, i) => {
      ctx.strokeStyle = i === this.highlight ? accent : ink;
      ctx.lineWidth = (i === this.highlight ? 4 : 3) * s;
      ctx.beginPath();
      stroke.forEach(([x, y], j) => (j ? ctx.lineTo(x * s, y * s) : ctx.moveTo(x * s, y * s)));
      if (stroke.length === 1) ctx.lineTo(stroke[0][0] * s + 0.1, stroke[0][1] * s);
      ctx.stroke();
    });
  }
}
