import { describe, expect, it } from "vitest";
import { mulberry32, sampleCategorical, sampleRow } from "../src/ghost";

const M = 3;

function params({ k = 1, mux = 2, muy = -1, logSigma = -9, pen = [5, -5, -5] } = {}) {
  const p = new Array(6 * M + 3).fill(0);
  for (let i = 0; i < M; i++) p[i] = i === k ? 20 : -20;
  p[M + k] = mux;
  p[2 * M + k] = muy;
  p[3 * M + k] = logSigma;
  p[4 * M + k] = logSigma;
  p.splice(6 * M, 3, ...pen);
  return p;
}

describe("MDN sampling", () => {
  it("draws near the mean of the dominant component", () => {
    const rand = mulberry32(1);
    for (let i = 0; i < 20; i++) {
      const s = sampleRow(params(), M, 0.4, rand);
      expect(s.dx).toBeCloseTo(2, 2);
      expect(s.dy).toBeCloseTo(-1, 2);
      expect(s.pen).toBe(0);
    }
  });

  it("spread grows with sigma and shrinks with temperature", () => {
    const spread = (temperature: number) => {
      const rand = mulberry32(2);
      const xs = Array.from({ length: 2000 }, () => sampleRow(params({ logSigma: 0 }), M, temperature, rand).dx);
      const mean = xs.reduce((a, b) => a + b) / xs.length;
      return Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length);
    };
    expect(spread(1)).toBeCloseTo(1, 1);
    expect(spread(0.25)).toBeCloseTo(0.5, 1);
  });

  it("categorical follows the logits", () => {
    const rand = mulberry32(3);
    const counts = [0, 0];
    for (let i = 0; i < 4000; i++) counts[sampleCategorical([Math.log(3), 0], 1, rand)]++;
    expect(counts[0] / 4000).toBeCloseTo(0.75, 1);
  });

  it("picks the end token when its logit dominates", () => {
    expect(sampleRow(params({ pen: [-9, -9, 9] }), M, 0.4, mulberry32(4)).pen).toBe(2);
  });
});
