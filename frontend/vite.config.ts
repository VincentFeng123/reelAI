import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: Vite serves the UI on :5173 and proxies API calls to FastAPI on :8000,
// so the browser sees same-origin. Prod: `vite build` emits into backend/static,
// which FastAPI serves itself (same-origin, no proxy).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/jobs": { target: "http://localhost:8000", changeOrigin: true },
      "/clips": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
  build: {
    outDir: "../backend/static",
    emptyOutDir: true,
  },
});
