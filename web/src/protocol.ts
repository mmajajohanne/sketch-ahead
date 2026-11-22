import type { Row } from "./preprocess";

/** Contents of public/model.json (written by train/export_onnx.py). */
export interface ModelMeta {
  classes: string[];
  temperature: number;
  canvas: number;
  origin: [number, number];
  offset_scale: number;
  sos: number[];
  max_len: number;
  hidden: number;
  layers: number;
  n_mix: number;
}

export type ToWorker =
  /** Forget the drawing: zero LSTM state, next stroke starts with SOS. */
  | { type: "reset" }
  /** Feed one finished stroke. `ghost`: also sample a continuation. */
  | { type: "stroke"; id: number; rows: Row[]; ghost: boolean };

export type FromWorker =
  | { type: "ready"; meta: ModelMeta; backend: string }
  | { type: "error"; message: string }
  | {
      type: "stroke";
      id: number;
      /** Calibrated class probabilities after each point of the stroke. */
      probs: number[][];
      /** Sampled continuation for the top class, offsets from the stroke's last point. */
      ghost: { cls: number; rows: Row[] } | null;
      ms: number;
    };
