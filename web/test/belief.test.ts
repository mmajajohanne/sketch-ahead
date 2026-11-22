import { describe, expect, it } from "vitest";
import { attribution, decide, entropy } from "../src/belief";

describe("decision with hysteresis", () => {
  it("decides when a class reaches ENTER and keeps the first stroke", () => {
    expect(decide([[0.5, 0.5], [0.75, 0.25], [0.9, 0.1]])).toEqual({ cls: 0, stroke: 1 });
  });

  it("stays decided between EXIT and ENTER", () => {
    expect(decide([[0.8, 0.2], [0.5, 0.5], [0.45, 0.55]])).toEqual({ cls: 0, stroke: 0 });
  });

  it("lets go below EXIT, and can switch to a new class", () => {
    expect(decide([[0.8, 0.2], [0.35, 0.65]])).toBeNull();
    expect(decide([[0.8, 0.2], [0.2, 0.8]])).toEqual({ cls: 1, stroke: 1 });
  });

  it("is null for no strokes", () => {
    expect(decide([])).toBeNull();
  });
});

describe("attribution and entropy", () => {
  it("Δp per stroke against a uniform prior", () => {
    const a = attribution([[0.6, 0.4], [0.3, 0.7], [0.1, 0.9]], 1);
    [-0.1, 0.3, 0.2].forEach((v, i) => expect(a[i]).toBeCloseTo(v, 9));
  });

  it("entropy in bits", () => {
    expect(entropy([0.5, 0.5])).toBeCloseTo(1);
    expect(entropy([1, 0, 0])).toBe(0);
  });
});
