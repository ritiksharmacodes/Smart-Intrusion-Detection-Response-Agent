import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import mkcert from "vite-plugin-mkcert";

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    mkcert()
  ],
  server: {
    host: true,         // <-- IMPORTANT
    port: 5173,         // optional, but consistent
    strictPort: true,   // optional
    https: true,
  }
})
