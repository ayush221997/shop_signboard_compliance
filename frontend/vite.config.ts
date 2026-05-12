import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiTarget = 'http://127.0.0.1:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    strictPort: false,
    proxy: {
      '/api': { target: apiTarget, changeOrigin: true },
      '/files': { target: apiTarget, changeOrigin: true },
    },
  },
  preview: {
    host: true,
    port: 4173,
    strictPort: false,
    proxy: {
      '/api': { target: apiTarget, changeOrigin: true },
      '/files': { target: apiTarget, changeOrigin: true },
    },
  },
})
