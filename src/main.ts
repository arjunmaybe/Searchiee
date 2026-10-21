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
const noticeText = document.querySelector<HTMLParagraphElement>("#notice-text")!;
const foldersEl = document.querySelector<HTMLUListElement>("#folders")!;
const foldersMeta = document.querySelector<HTMLParagraphElement>("#folders-meta")!;
const resultsMeta = document.querySelector<HTMLParagraphElement>("#results-meta")!;

let debounceTimer: ReturnType<typeof setTimeout>;
let searchRequestId = 0;

queryInput.addEventListener("input", () => {
  searchRequestId += 1;
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
  foldersEl.replaceChildren();
  foldersMeta.textContent =
    folders.length === 0
      ? "No folders indexed"
      : formatCount(folders.length, "folder", "folders");
  foldersMeta.title = foldersMeta.textContent;

  for (const folder of folders) {
    const parts = splitPath(folder.path);

    const row = document.createElement("li");
    row.className = "folder-row";

    const identity = document.createElement("div");
    identity.className = "folder-identity";

    const nameEl = document.createElement("span");
    nameEl.className = "folder-name";
    nameEl.textContent = parts.name || folder.path;
    nameEl.title = folder.path;

    const pathEl = document.createElement("span");
    pathEl.className = "folder-path";
    pathEl.textContent = parts.parent;
    pathEl.title = folder.path;
    pathEl.hidden = !parts.parent;

    identity.append(nameEl, pathEl);

    const countEl = document.createElement("span");
    countEl.className = "folder-count";
    countEl.textContent = formatCount(folder.file_count, "file", "files");

    const actions = document.createElement("div");
    actions.className = "folder-actions";

    const refreshBtn = document.createElement("button");
    refreshBtn.type = "button";
    refreshBtn.className = "button button-secondary";
    refreshBtn.textContent = "Refresh";
    refreshBtn.setAttribute("aria-label", `Refresh ${folder.path}`);
    refreshBtn.addEventListener("click", () =>
      void refreshFolder(folder.path, refreshBtn),
    );

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "button button-danger";
    removeBtn.textContent = "Remove";
    removeBtn.setAttribute("aria-label", `Remove ${folder.path} from index`);
    removeBtn.addEventListener("click", () =>
      void removeFolder(folder.path, removeBtn),
    );

    actions.append(refreshBtn, removeBtn);
    row.append(identity, countEl, actions);
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
  const requestId = ++searchRequestId;
  const query = queryInput.value.trim();
  if (!query) {
    resultsEl.replaceChildren();
    resultsMeta.textContent = "";
    resultsMeta.title = "";
    return;
  }

  const results = await invoke<SearchResult[]>("search", { query });
  if (requestId !== searchRequestId) return;
  render(results, query);
}

function render(results: SearchResult[], query: string) {
  resultsEl.replaceChildren();

  resultsMeta.textContent = `${formatCount(results.length, "result", "results")} for “${query}”`;
  resultsMeta.title = resultsMeta.textContent;

  if (results.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state no-results";

    const title = document.createElement("p");
    title.className = "empty-title";
    title.textContent = "No sufficiently relevant results";

    empty.append(title);
    resultsEl.append(empty);
    return;
  }

  for (const result of results) {
    const parts = splitPath(result.path);

    const article = document.createElement("article");
    article.className = "result";

    const head = document.createElement("div");
    head.className = "result-head";

    const identity = document.createElement("div");
    identity.className = "result-identity";

    const nameEl = document.createElement("h3");
    nameEl.className = "result-name";
    nameEl.textContent = parts.name || result.path;
    nameEl.title = result.path;

    const pathEl = document.createElement("p");
    pathEl.className = "result-path";
    pathEl.textContent = parts.parent;
    pathEl.title = result.path;
    pathEl.hidden = !parts.parent;

    identity.append(nameEl, pathEl);

    const scoreEl = document.createElement("span");
    scoreEl.className = "result-score";
    scoreEl.textContent = `Similarity ${(result.score * 100).toFixed(1)}%`;
    scoreEl.title = "Cosine similarity";

    head.append(identity, scoreEl);

    const snippetEl = document.createElement("p");
    snippetEl.className = "result-snippet";
    snippetEl.textContent = result.snippet;

    article.append(head, snippetEl);
    resultsEl.append(article);
  }
}

function splitPath(path: string): { name: string; parent: string } {
  if (/^[A-Za-z]:[\\/]$/.test(path)) {
    return { name: path, parent: "" };
  }

  const normalized = path.replace(/[\\/]+$/, "");
  const separator = Math.max(
    normalized.lastIndexOf("\\"),
    normalized.lastIndexOf("/"),
  );
  if (separator < 0) {
    return { name: normalized || path, parent: "" };
  }
  const parent = normalized.slice(0, separator);
  return {
    name: normalized.slice(separator + 1),
    parent: /^[A-Za-z]:$/.test(parent)
      ? `${parent}${normalized[separator]}`
      : parent,
  };
}

function formatCount(
  count: number,
  singular: string,
  plural: string,
): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

void (async () => {
  await loadFolders();
  try {
    const notice = await invoke<string | null>("get_startup_notice");
    if (notice) {
      noticeEl.hidden = false;
      noticeText.textContent = notice;
      noticeEl.addEventListener("click", () => {
        noticeEl.hidden = true;
      });
    }
  } catch {
    // Stay quiet; search remains usable without the notice.
  }
})();
