import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
    plugins: [react()],
    server: {
        port: 5173,
        proxy: {
            "/api": "http://127.0.0.1:8000",
            "/assets": "http://127.0.0.1:8000"
        }
    },
    build: {
        outDir: "dist",
        emptyOutDir: true,
        chunkSizeWarningLimit: 1200,
        rollupOptions: {
            output: {
                manualChunks: {
                    react: ["react", "react-dom"],
                    three: ["three", "@react-three/fiber", "@react-three/drei"]
                }
            }
        }
    }
});
