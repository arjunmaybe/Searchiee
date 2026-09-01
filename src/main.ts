import { invoke } from "@tauri-apps/api/core";

interface SearchResult {
  path: string;
  snippet: string;
  score: number;
}

const queryInput = document.querySelector<HTMLInputElement>("#query")!;
const resultsEl = document.querySelector<HTMLDivElement>("#results")!;
const indexBtn = document.querySelector<HTMLButtonElement>("#index-btn")!;

let debounceTimer: ReturnType<typeof setTimeout>;

queryInput.addEventListener("input", () => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(runSearch, 250);
});

indexBtn.addEventListener("click", async () => {
  // v0: hardcode a folder path for now. Swap in Tauri's file/folder
  // picker dialog once the core loop works.
  const folder = prompt("Folder path to index:");
  if (!folder) return;

  indexBtn.textContent = "Indexing...";
  const count = await invoke<number>("index_folder", { path: folder });
  indexBtn.textContent = "Index a folder";
  alert(`Indexed ${count} files`);
});

async function runSearch() {
  const query = queryInput.value.trim();
  if (!query) {
    resultsEl.innerHTML = "";
    return;
  }

  const results = await invoke<SearchResult[]>("search", { query });
  render(results);
}

function render(results: SearchResult[]) {
  resultsEl.innerHTML = results
    .map(
      (r) => `
      <div class="result">
        <div class="path">${escapeHtml(r.path)} — ${(r.score * 100).toFixed(1)}%</div>
        <div class="snippet">${escapeHtml(r.snippet)}</div>
      </div>
    `,
    )
    .join("");
}

function escapeHtml(s: string): string {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}
