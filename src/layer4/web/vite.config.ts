import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    target: "es2022",
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    host: "0.0.0.0",
    port: 8080,
    proxy: {
      "/health": "http://127.0.0.1:8090",
      "/train": "http://127.0.0.1:8090",
      "/settings": "http://127.0.0.1:8090",
      "/telemetry": "http://127.0.0.1:8090",
      "/maps": "http://127.0.0.1:8090",
      "/ws": { target: "ws://127.0.0.1:8090", ws: true },
    },
  },
});
