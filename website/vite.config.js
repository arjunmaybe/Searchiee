import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const page = (path) => fileURLToPath(new URL(path, import.meta.url));

export default defineConfig({
  appType: "mpa",
  server: {
    port: 5174,
    strictPort: true,
  },
  preview: {
    port: 4174,
    strictPort: true,
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        home: page("./index.html"),
        privacy: page("./privacy/index.html"),
        notFound: page("./404.html"),
      },
    },
  },
});
