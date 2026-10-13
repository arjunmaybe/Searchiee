import { invoke } from "@tauri-apps/api/core";

interface SearchResult {
  path: string;
  snippet: string;
  score: number;
}

interface FolderInfo {
  path: string;
  file_count: number;
}

const queryInput = document.querySelector<HTMLInputElement>("#query")!;
const resultsEl = document.querySelector<HTMLDivElement>("#results")!;
const indexBtn = document.querySelector<HTMLButtonElement>("#index-btn")!;
const noticeEl = document.querySelector<HTMLDivElement>("#notice")!;
const foldersEl = document.querySelector<HTMLDivElement>("#folders")!;

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
    await loadFolders();
    await runSearch();
  }
});

async function loadFolders() {
  let folders: FolderInfo[];
  try {
    folders = await invoke<FolderInfo[]>("list_folders");
  } catch {
    return;
  }
  foldersEl.innerHTML = "";
  for (const folder of folders) {
    const row = document.createElement("div");
    row.className = "folder-row";

    const pathEl = document.createElement("span");
    pathEl.className = "folder-path";
    pathEl.textContent = folder.path;
    pathEl.title = folder.path;

    const countEl = document.createElement("span");
    countEl.className = "folder-count";
    countEl.textContent = `${folder.file_count} files`;

    const refreshBtn = document.createElement("button");
    refreshBtn.textContent = "Refresh";
    refreshBtn.addEventListener("click", () => void refreshFolder(folder.path, refreshBtn));

    const removeBtn = document.createElement("button");
    removeBtn.textContent = "Remove";
    removeBtn.addEventListener("click", () => void removeFolder(folder.path, removeBtn));

    row.append(pathEl, countEl, refreshBtn, removeBtn);
    foldersEl.append(row);
  }
}

async function refreshFolder(path: string, btn: HTMLButtonElement) {
  btn.textContent = "Refreshing...";
  try {
    const count = await invoke<number>("refresh_folder", { path });
    alert(`Refreshed ${count} files`);
  } catch (e) {
    alert(`Refresh failed: ${e instanceof Error ? e.message : String(e)}`);
  } finally {
    await loadFolders();
    await runSearch();
  }
}

async function removeFolder(path: string, btn: HTMLButtonElement) {
  const confirmed = confirm(
    `Remove this folder from the index?\n\n${path}\n\nOnly index entries are removed; original files stay on disk.`,
  );
  if (!confirmed) return;
  btn.textContent = "Removing...";
  try {
    const removed = await invoke<number>("remove_folder", { path });
    alert(`Removed ${removed} indexed files`);
  } catch (e) {
    alert(`Remove failed: ${e instanceof Error ? e.message : String(e)}`);
  } finally {
    await loadFolders();
    await runSearch();
  }
}

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
  await loadFolders();
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
