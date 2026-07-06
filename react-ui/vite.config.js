import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiPort = process.env.ROOP_API_PORT || '8001'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${apiPort}`,
        changeOrigin: true,
      }
    }
  }
})
