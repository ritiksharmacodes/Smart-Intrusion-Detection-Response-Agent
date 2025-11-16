import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],

  // ✔ Allow loading ONNX wasm files correctly
  assetsInclude: ['**/*.wasm'],

  // ✔ Prevent Vite from pre-bundling ONNX Runtime Web
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },

  // ✔ Ensure wasm files are served with proper MIME type
  server: {
    mimeTypes: {
      'application/wasm': ['wasm']
    }
  }
})
