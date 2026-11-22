# web/

Browser app: Vite + TypeScript, model runs with ONNX Runtime Web (WASM) in
a Web Worker. No server-side code; `npm run build` gives a static site.

```bash
cd web
npm install
npm run dev     # http://localhost:5173
npm test        # vitest: preprocessing parity, MDN sampling, belief logic
npm run build   # typecheck + static build in web/dist/
```

`public/classifier.onnx`, `public/generator.onnx` and `public/model.json` are written by
`train/export_onnx.py`.

| File | Role |
|---|---|
| `src/canvas.ts` | Pointer input mapped to canvas units 0–255, renders strokes and the dashed ghost |
| `src/preprocess.ts` | RDP simplification (ε = 2, as Quick, Draw!) and stroke-5 rows. Must match `train/data/strokes.py` **exactly**: `test/preprocess.test.ts` checks it against a fixture that `train/tests/test_web_fixture.py` checks against Python |
| `src/worker.ts` | Runs `classifier.onnx` statefully (each finished stroke fed once) and `generator.onnx` for the ghost: replayed over the drawing for the top class, then sampled one point at a time (temperature 0.25, one stroke, only when the top class is at least 60 %, resampled if it leaves the canvas) |
| `src/ghost.ts` | Sampling from the MDN head, same as `sample_continuation` in `train/model.py` |
| `src/belief.ts` | Belief chart, decision with hysteresis (enter at 70 %, let go below 40 %), stroke attribution (Δp of the top class per stroke), entropy |
| `src/protocol.ts` | Message types between page and worker, and the `model.json` schema |
| `src/main.ts` | Wires it together; undo and clear replay the remaining strokes from a fresh state |

The model updates on pointer-up, once per stroke: the training data is
RDP-simplified per stroke, so a half-finished stroke has no counterpart in
training.
