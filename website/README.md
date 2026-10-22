# Searchiee website

Public-facing site for Searchiee, a local-first Windows desktop app for
semantic search across Markdown and plain-text files.

This is an independent Vite multi-page static site. It does not use a
frontend framework, a client-side router, a backend, analytics, or
third-party runtime scripts. The desktop application and its root package
are not part of this build.

## Local development

Run these commands from this directory:

```powershell
npm ci
npm run dev
```

The development server runs at `http://localhost:5174`. Port 5174 is strict
so the site cannot silently collide with the Tauri app's Vite server on
port 5173.

## Production build and preview

```powershell
npm ci
npm run build
npm run preview
```

`npm run build` writes to `website/dist/`. The local production preview runs
at `http://localhost:4174` with a strict port.

Do not run these commands from the repository root. The website has its own
package, lockfile, dependencies, and build output.

## Structure

```text
website/
├── package.json
├── package-lock.json
├── vite.config.js
├── README.md
├── index.html
├── privacy/
│   └── index.html
├── 404.html
├── src/
│   └── styles.css
└── public/
    └── favicon.ico
```

`vite.config.js` declares three HTML inputs: the landing page, privacy page,
and 404 page. The build output is independent of the root Tauri application.

## Design and implementation decisions

- The visual direction is an editorial document index: warm paper, ink,
  hairline rules, a deep-pine accent, and numbered section labels.
- Typography uses the desktop app's system font families: Iowan
  Old Style/Palatino/Georgia for display, Segoe UI for text, and Cascadia
  Code/Consolas for technical labels. No webfonts are downloaded.
- Spacing follows a 4px scale. The shell is 72rem and prose is capped at
  68ch. Corners are 6px; there are no shadows or gradients.
- The landing page does not depend on an application screenshot. It uses a
  factual index card rather than a fabricated product interface.
- Native HTML is the entire interaction model. There is no client-side
  JavaScript. The JSON-LD block is structured data, not application code.
- The site is intentionally light: no icons, decorative animation, pricing,
  testimonials, newsletter form, or fake product counts.
- The 404 page is a real page in the build. The hosting provider must return
  an HTTP 404 status and serve `404.html` for unknown paths. A built
  `404.html` alone does not guarantee correct status behavior.

## Verified product facts and links

Last verified: 22 October 2026.

- Repository: https://github.com/arjunmaybe/Searchiee
- Release page: https://github.com/arjunmaybe/Searchiee/releases/tag/v0.1.0
- Windows NSIS installer:
  https://github.com/arjunmaybe/Searchiee/releases/download/v0.1.0/Searchiee_0.1.0_x64-setup.exe
- Windows MSI installer:
  https://github.com/arjunmaybe/Searchiee/releases/download/v0.1.0/Searchiee_0.1.0_x64_en-US.msi
- Checksums:
  https://github.com/arjunmaybe/Searchiee/releases/download/v0.1.0/SHA256SUMS.txt
- Issues: https://github.com/arjunmaybe/Searchiee/issues

Product facts used on the site (downloadable v0.1.0):

- Searchiee is a Windows 10/11 x64 desktop app, not a hosted service.
- It indexes only Markdown (`.md`) and plain-text (`.txt`) files.
- The user adds and refreshes folders manually; there is no background
  filesystem watcher.
- Embeddings are generated locally with FastEmbed `AllMiniLML6V2`
  (384 dimensions).
- In v0.1.0, each selected Markdown or text file is embedded as a whole;
  documents are not split into chunks.
- The local index stores paths, document contents, and embeddings in
  `index-v1.json` inside Tauri's application-specific local data directory.
- Search returns one result per file, up to 10, with a snippet of up to
  200 characters and a cosine similarity score.
- v0.1.0 ranks files by cosine similarity without a fixed minimum-score
  filter.
- The first model download is roughly 90 MB and requires network access.
  Later indexing and search work offline once the model is cached.
- v0.1.0 contains no telemetry, account system, or cloud search service.
- Release v0.1.0 was published on 13 October 2026.

The site does not call the project open source. The repository currently has
no license file, and GitHub reports no declared license.

### Post-release changes in the current source

The current source contains changes made after v0.1.0 was published, including
long-document chunking and a `0.25` minimum cosine-similarity threshold. These
changes are not part of the downloadable v0.1.0 installer. Public copy that
describes v0.1.0 must not present chunking, best-matching chunks, or a fixed
threshold as release behavior.

When a future release includes those changes, update the release links and the
release-specific copy together.

## Missing product image

There is no suitable high-resolution application screenshot in the
repository. The existing app icon (`src-tauri/icons/icon.ico`) is a multi-size
ICO containing 16, 24, 32, 48, 64, 128, and 256 pixel frames. The site uses the
same file as its favicon, linked at 32×32. It is not a product screenshot. Do
not substitute a fabricated interface or a broken placeholder.

The exact screenshot to capture later:

1. Run Searchiee 0.1.0 from an isolated synthetic demo library. Do not use a
   real saved index, personal paths, or personal document contents.
2. Use a clean search-results state at 1440×1000 logical pixels, 100% Windows
   scaling (or capture at 2× for high-DPI and downscale). Hide the startup
   notice, folder prompt, alerts, and cursor focus if possible.
3. Show a natural-language query, at least one real result from the demo
   library, its file name, snippet, and similarity score. Keep the registered
   folder list visible if it fits without crowding.
4. Use fictitious paths such as `D:\Demo\release-notes` and synthetic notes
   only. Do not include the user's normal folders or file names.
5. Save the approved PNG under `website/public/screenshots/` and export a
   responsive WebP around 960px wide. Add descriptive alt text and a caption
   that explicitly says it is a demo library.
6. Add the image only after it has been reviewed. The current hero must keep
   working without it; do not introduce an empty image frame.

A real 1200×630 Open Graph image should be produced later from the approved
screenshot and wordmark. Do not generate a fake UI for it.

## Domain-dependent launch work

The production domain is intentionally deferred. Do not deploy this draft as
production while these tasks are unresolved. Do not use localhost, a preview
URL, or a guessed domain as a canonical.

Before launch, with the verified production origin:

- Add a self-referencing canonical URL to every indexable page.
- Add `og:url` and an absolute `og:image` URL, and switch the Twitter card to
  `summary_large_image` if a real social image is ready.
- Add `robots.txt` and `sitemap.xml` with absolute production URLs.
- Configure preview deployments to send `noindex` so previews cannot be
  indexed as production.
- Verify HTTPS, canonical links, Open Graph previews, and the 404 status after
  deployment.
- Recheck the installer, checksums, release, and issue URLs at launch.

The current HTML intentionally omits canonical, `og:url`, `og:image`, and
sitemap files. It does not contain fake placeholders.

## Legal and factual details still requiring the owner

- Choose and add a license if the project should be described as open source.
- Decide whether a dedicated privacy contact email and legal entity name are
  required for the jurisdictions where the site and app are offered.
- Confirm whether a separate terms page or installer EULA is needed.
- Approve the privacy wording after the production hosting provider is chosen,
  because hosting request logs are controlled by that provider.

The privacy page currently identifies GitHub Issues as the verified contact
channel and states that a privacy email or legal entity is not published.

## Hosting notes

No deployment provider is configured in this repository. Any static host can
serve `website/dist/`; options include Cloudflare Pages, Netlify, or GitHub
Pages with a custom domain.

- Build command: `npm ci && npm run build` from `website/`
- Publish directory: `website/dist`
- Node.js: 20 or newer
- Configure a real 404 response for unknown paths.
- Set security headers at the host if supported. Keep the site free of
  third-party runtime requests.
- Do not add analytics without a concrete measurement need and an accurate
  privacy disclosure.

## Scope safeguards

- All website work belongs under `website/`.
- Do not change the root Searchiee package, lockfile, Tauri configuration,
  desktop behavior, indexing, persistence, ranking, or the relevance threshold
  that exists only in the post-v0.1.0 source.
- Do not touch `clippy-out.txt`, evaluation outputs, `test-out.txt`, or
  `src-tauri/.fastembed_cache/`.
- Do not add CI workflows or deploy from this task without separate approval.
