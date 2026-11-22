import { describe, expect, it } from "vitest";
import fixture from "./fixtures/preprocess.json";
import { drawingToRows, rdp, rowsToPoints, simplifyStroke, strokeToRows, type Pt } from "../src/preprocess";

describe("preprocess matches train/data/strokes.py", () => {
  for (const [i, c] of fixture.cases.entries()) {
    it(`fixture drawing ${i}`, () => {
      const rows = drawingToRows(c.drawing as [number[], number[]][]);
      expect(rows.length).toBe(c.rows.length);
      rows.forEach((r, j) => r.forEach((v, k) => expect(v).toBeCloseTo(c.rows[j][k], 5)));
    });
  }

  it("stroke by stroke gives the same rows as the whole drawing", () => {
    const d = fixture.cases[2].drawing as [number[], number[]][];
    let prev: Pt | null = null;
    const rows = d.flatMap(([xs, ys]) => {
      const out = strokeToRows(xs.map((x, i) => [x, ys[i]] as Pt), prev);
      prev = out.last;
      return out.rows;
    });
    expect(rows).toEqual(drawingToRows(d));
  });

  it("rowsToPoints inverts strokeToRows", () => {
    const stroke: Pt[] = [[10, 20], [30, 40], [100, 5]];
    const pts = rowsToPoints(strokeToRows(stroke, null).rows, [0.5, 0.5]);
    pts.forEach((p, i) => {
      expect(p.x * 256).toBeCloseTo(stroke[i][0], 6);
      expect(p.y * 256).toBeCloseTo(stroke[i][1], 6);
    });
    expect(pts.map((p) => p.lift)).toEqual([false, false, true]);
  });
});

describe("simplification", () => {
  it("rdp drops points on a straight line", () => {
    const line: Pt[] = Array.from({ length: 50 }, (_, i) => [i, 2 * i]);
    expect(rdp(line)).toEqual([[0, 0], [49, 98]]);
  });

  it("rdp keeps a corner", () => {
    const corner: Pt[] = [...Array.from({ length: 20 }, (_, i) => [i, 0] as Pt), ...Array.from({ length: 20 }, (_, i) => [19, i + 1] as Pt)];
    expect(rdp(corner)).toEqual([[0, 0], [19, 0], [19, 20]]);
  });

  it("simplifyStroke rounds, clamps and removes duplicates", () => {
    expect(simplifyStroke([[-5, 10.4], [-3, 10.2], [300, 10.6]])).toEqual([[0, 10], [255, 11]]);
    expect(simplifyStroke([[5, 5], [5.2, 5.1]])).toEqual([[5, 5]]);
  });
});
