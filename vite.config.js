import { defineConfig } from "vite";

// Relative asset paths so the built site works at https://<user>.github.io/<repo>/
export default defineConfig({
  base: "./",
});
