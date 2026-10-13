import { invoke } from "@tauri-apps/api/core";

interface SearchResult {
  path: string;
  snippet: string;
  score: number;
}

const queryInput = document.querySelector<HTMLInputElement>("#query")!;
const resultsEl = document.querySelector<HTMLDivElement>("#results")!;
const indexBtn = document.querySelector<HTMLButtonElement>("#index-btn")!;
const noticeEl = document.querySelector<HTMLDivElement>("#notice")!;

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
  try {
    const count = await invoke<number>("index_folder", { path: folder });
    alert(`Indexed ${count} files`);
  } catch (e) {
    alert(`Indexing failed: ${e instanceof Error ? e.message : String(e)}`);
  } finally {
    indexBtn.textContent = "Index a folder";
  }
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

void (async () => {
  try {
    const notice = await invoke<string | null>("get_startup_notice");
    if (notice) {
      noticeEl.hidden = false;
      noticeEl.textContent = notice;
      noticeEl.addEventListener("click", () => {
        noticeEl.hidden = true;
      });
    }
  } catch {
    // Stay quiet; search remains usable without the notice.
  }
})();
