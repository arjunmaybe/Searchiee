// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

interface SearchResult {
  path: string;
  snippet: string;
  score: number;
}

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
}

const invokeMock = vi.hoisted(() => vi.fn());

vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));

const fixture = `
  <input id="query" type="text" />
  <button id="index-btn" type="button">Index a folder</button>
  <div id="notice" hidden><p id="notice-text"></p></div>
  <ul id="folders"></ul>
  <p id="folders-meta"></p>
  <div id="results"></div>
  <p id="results-meta"></p>
`;

let searchCalls: string[];
let searches: Map<string, Deferred<SearchResult[]>>;

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function getElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Missing test fixture element: ${selector}`);
  }
  return element;
}

async function flushMicrotasks() {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}

async function typeQuery(value: string) {
  const query = getElement<HTMLInputElement>("#query");
  query.value = value;
  query.dispatchEvent(new Event("input"));
  await vi.advanceTimersByTimeAsync(250);
  await flushMicrotasks();
}

function searchDeferred(query: string): Deferred<SearchResult[]> {
  const search = searches.get(query);
  if (!search) {
    throw new Error(`No pending search for query: ${query}`);
  }
  return search;
}

describe("search error handling", () => {
  beforeEach(async () => {
    vi.resetModules();
    document.body.innerHTML = fixture;
    searchCalls = [];
    searches = new Map();

    invokeMock.mockReset();
    invokeMock.mockImplementation(
      (command: string, args?: { query?: string }) => {
        if (command === "list_folders") {
          return Promise.resolve([]);
        }
        if (command === "get_startup_notice") {
          return Promise.resolve(null);
        }
        if (command === "search") {
          const query = args?.query ?? "";
          searchCalls.push(query);
          const search = deferred<SearchResult[]>();
          searches.set(query, search);
          return search.promise;
        }
        return Promise.reject(new Error(`Unexpected command: ${command}`));
      },
    );

    await import("./main");
    await flushMicrotasks();

    vi.useFakeTimers();
  });

  afterEach(async () => {
    vi.clearAllTimers();
    for (const search of searches.values()) {
      search.resolve([]);
    }
    await flushMicrotasks();
    vi.useRealTimers();
    document.body.innerHTML = "";
    invokeMock.mockReset();
  });

  it("renders a readable, escaped error for a rejected search", async () => {
    const message = 'Search backend unavailable: <model> & "index"';

    await typeQuery("broken");
    searchDeferred("broken").reject(message);
    await flushMicrotasks();

    const results = getElement<HTMLDivElement>("#results");
    const meta = getElement<HTMLParagraphElement>("#results-meta");

    expect(results.querySelector(".empty-title")?.textContent).toBe(
      "Search failed",
    );
    expect(results.querySelector(".empty-copy")?.textContent).toBe(message);
    expect(results.textContent).toContain(message);
    expect(results.querySelector("model")).toBeNull();
    expect(meta.textContent).toBe("Search failed for “broken”");
    expect(meta.title).toBe(meta.textContent);
    expect(results.querySelector(".result")).toBeNull();
  });

  it("clears results from an earlier query when a new search fails", async () => {
    await typeQuery("alpha");
    searchDeferred("alpha").resolve([
      {
        path: "C:\\docs\\alpha.md",
        snippet: "alpha match",
        score: 0.9,
      },
    ]);
    await flushMicrotasks();

    const results = getElement<HTMLDivElement>("#results");
    const meta = getElement<HTMLParagraphElement>("#results-meta");

    expect(results.querySelectorAll(".result")).toHaveLength(1);
    expect(meta.textContent).toBe("1 result for “alpha”");

    await typeQuery("beta");
    searchDeferred("beta").reject(new Error("index unavailable"));
    await flushMicrotasks();

    expect(results.querySelectorAll(".result")).toHaveLength(0);
    expect(results.textContent).toContain("Search failed");
    expect(results.textContent).toContain("index unavailable");
    expect(meta.textContent).toBe("Search failed for “beta”");
    expect(searchCalls).toEqual(["alpha", "beta"]);
  });

  it("ignores a stale failure after a newer query becomes active", async () => {
    await typeQuery("alpha");
    const alpha = searchDeferred("alpha");

    await typeQuery("beta");
    const beta = searchDeferred("beta");

    beta.resolve([
      {
        path: "C:\\docs\\beta.md",
        snippet: "beta match",
        score: 0.8,
      },
    ]);
    await flushMicrotasks();

    const results = getElement<HTMLDivElement>("#results");
    const meta = getElement<HTMLParagraphElement>("#results-meta");

    expect(results.querySelector(".result-name")?.textContent).toBe("beta.md");
    expect(meta.textContent).toBe("1 result for “beta”");

    alpha.reject("stale failure");
    await flushMicrotasks();

    expect(results.querySelectorAll(".result")).toHaveLength(1);
    expect(results.querySelector(".result-name")?.textContent).toBe("beta.md");
    expect(meta.textContent).toBe("1 result for “beta”");
    expect(results.textContent).not.toContain("Search failed");
  });

  it("invalidates a pending search and clears results when the input is cleared", async () => {
    await typeQuery("seed");
    searchDeferred("seed").resolve([
      {
        path: "C:\\docs\\seed.md",
        snippet: "seed match",
        score: 0.7,
      },
    ]);
    await flushMicrotasks();

    const results = getElement<HTMLDivElement>("#results");
    const meta = getElement<HTMLParagraphElement>("#results-meta");
    expect(results.querySelectorAll(".result")).toHaveLength(1);

    await typeQuery("alpha");
    const alpha = searchDeferred("alpha");

    const query = getElement<HTMLInputElement>("#query");
    query.value = "";
    query.dispatchEvent(new Event("input"));
    await vi.advanceTimersByTimeAsync(250);
    await flushMicrotasks();

    expect(results.innerHTML).toBe("");
    expect(meta.textContent).toBe("");
    expect(meta.title).toBe("");
    expect(searchCalls).toEqual(["seed", "alpha"]);

    alpha.resolve([
      {
        path: "C:\\docs\\alpha.md",
        snippet: "late alpha match",
        score: 0.6,
      },
    ]);
    await flushMicrotasks();

    expect(results.innerHTML).toBe("");
    expect(meta.textContent).toBe("");
    expect(meta.title).toBe("");
  });

  it("treats whitespace-only input like an empty query", async () => {
    await typeQuery("seed");
    searchDeferred("seed").resolve([
      {
        path: "C:\\docs\\seed.md",
        snippet: "seed match",
        score: 0.7,
      },
    ]);
    await flushMicrotasks();

    const results = getElement<HTMLDivElement>("#results");
    const meta = getElement<HTMLParagraphElement>("#results-meta");
    expect(results.querySelectorAll(".result")).toHaveLength(1);

    await typeQuery("   ");

    expect(searchCalls).toEqual(["seed"]);
    expect(results.innerHTML).toBe("");
    expect(meta.textContent).toBe("");
    expect(meta.title).toBe("");
  });

  it("renders filename, snippet, similarity score, and result count on success", async () => {
    await typeQuery("note");
    searchDeferred("note").resolve([
      {
        path: "C:\\docs\\note.md",
        snippet: "The matching snippet",
        score: 0.875,
      },
    ]);
    await flushMicrotasks();

    const results = getElement<HTMLDivElement>("#results");
    const meta = getElement<HTMLParagraphElement>("#results-meta");

    expect(results.querySelectorAll(".result")).toHaveLength(1);
    expect(results.querySelector(".result-name")?.textContent).toBe("note.md");
    expect(results.querySelector(".result-path")?.textContent).toBe(
      "C:\\docs",
    );
    expect(results.querySelector(".result-snippet")?.textContent).toBe(
      "The matching snippet",
    );
    expect(results.querySelector(".result-score")?.textContent).toBe(
      "Similarity 87.5%",
    );
    expect(meta.textContent).toBe("1 result for “note”");
    expect(results.textContent).not.toContain("Search failed");
  });
});
