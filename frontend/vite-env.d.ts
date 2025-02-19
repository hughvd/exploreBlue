/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_BACKEND_URL: string; // Define your environment variable
  // Add others here if needed (e.g., VITE_API_KEY)
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}