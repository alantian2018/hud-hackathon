import {defineConfig} from "vite";

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        index: "index.html",
        greedy: "greedy.html",
        rl: "rl.html",
        compare: "compare.html"
      }
    }
  }
});
