import { defineConfig } from "vite";

export default defineConfig({
  // Relative asset paths, so the build works from any sub-path (e.g. GitHub Pages).
  base: "./",
  // onnxruntime-web loads its .wasm next to its own module; pre-bundling breaks that.
  optimizeDeps: { exclude: ["onnxruntime-web", "onnxruntime-web/wasm"] },
  worker: { format: "es" },
});
