/**
 * Sampling from the MDN head, as `sample_continuation` in train/model.py.
 *
 * Layout of one MDN row (M = n_mix): pi logits [0, M), mu_x [M, 2M),
 * mu_y [2M, 3M), log sigma_x [3M, 4M), log sigma_y [4M, 5M), raw rho [5M, 6M),
 * pen logits (down, lift, end) [6M, 6M+3).
 */

export const PEN_DOWN = 0;
export const PEN_LIFT = 1;
export const PEN_END = 2;

export interface Sample {
  dx: number;
  dy: number;
  pen: number;
}

export type Rand = () => number;

/** Index drawn from softmax(logits / temperature). */
export function sampleCategorical(logits: ArrayLike<number>, temperature: number, rand: Rand): number {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i] / temperature);
  const w = Array.from(logits, (l) => Math.exp(l / temperature - max));
  let u = rand() * w.reduce((a, b) => a + b, 0);
  for (let i = 0; i < w.length; i++) {
    u -= w[i];
    if (u <= 0) return i;
  }
  return w.length - 1;
}

function gaussianPair(rand: Rand): [number, number] {
  const u1 = Math.max(rand(), 1e-12);
  const r = Math.sqrt(-2 * Math.log(u1));
  const t = 2 * Math.PI * rand();
  return [r * Math.cos(t), r * Math.sin(t)];
}

/** One next point from MDN params `p` (a 6M+3 row). */
export function sampleRow(p: ArrayLike<number>, nMix: number, temperature: number, rand: Rand = Math.random): Sample {
  const M = nMix;
  const pi = Array.from({ length: M }, (_, i) => p[i]);
  const k = sampleCategorical(pi, temperature, rand);
  const s = Math.sqrt(temperature);
  const sx = Math.exp(p[3 * M + k]) * s;
  const sy = Math.exp(p[4 * M + k]) * s;
  const rho = Math.min(0.999, Math.max(-0.999, Math.tanh(p[5 * M + k])));
  const [z1, z2] = gaussianPair(rand);
  const dx = p[M + k] + sx * z1;
  const dy = p[2 * M + k] + sy * (rho * z1 + Math.sqrt(1 - rho * rho) * z2);
  const pen = sampleCategorical([p[6 * M], p[6 * M + 1], p[6 * M + 2]], temperature, rand);
  return { dx, dy, pen };
}

/** Small seedable PRNG (mulberry32), so ghosts can be reproduced in tests. */
export function mulberry32(seed: number): Rand {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
