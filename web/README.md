# web/

Browser app (milestones 4–5). Not scaffolded yet: it will be created with
Vite + TypeScript once a trained model can be exported to ONNX.

Planned files in `src/`:

| File | Role |
|---|---|
| `canvas.ts` | Input capture with pointer events, maps the canvas to the 0–255 range |
| `preprocess.ts` | Must reproduce `train/data/strokes.py` **exactly** (normalize by canvas size, stroke-5, first point offset from centre) |
| `worker.ts` | ONNX Runtime Web inference, stateful: `(point, h, c) → (class_logits, mdn_params, h', c')`, plus ghost sampling |
| `belief.ts` | Belief chart, entropy, hysteresis commit, stroke attribution (Δp per stroke) |
| `ghost.ts` | Renders sampled completions at low opacity |

`public/model.onnx` is produced by `train/export_onnx.py` (to be written).
