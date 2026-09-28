import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the read-only API served by `python -m arena.dashboard_api`.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8787" } },
  build: { outDir: "dist", chunkSizeWarningLimit: 900 },
});
