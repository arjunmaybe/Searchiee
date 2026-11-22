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
├── licensing/
│   └── index.html
├── 404.html
├── src/
│   └── styles.css
└── public/
    └── favicon.ico
```

`vite.config.js` declares four HTML inputs: the landing page, privacy page,
licensing page, and 404 page. The build output is independent of the root Tauri application.

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

Last verified: 22 November 2026.

- Repository: https://github.com/arjunmaybe/Searchiee
- Current release: https://github.com/arjunmaybe/Searchiee/releases/tag/v0.2.0
- Windows NSIS installer:
  https://github.com/arjunmaybe/Searchiee/releases/download/v0.2.0/Searchiee_0.2.0_x64-setup.exe
- Windows MSI installer:
  https://github.com/arjunmaybe/Searchiee/releases/download/v0.2.0/Searchiee_0.2.0_x64_en-US.msi
- Checksums:
  https://github.com/arjunmaybe/Searchiee/releases/download/v0.2.0/SHA256SUMS.txt
- License summary: https://searchiee.pages.dev/licensing/
- Full license: https://github.com/arjunmaybe/Searchiee/blob/main/LICENSE
- Commercial licensing: https://github.com/arjunmaybe/Searchiee/blob/main/COMMERCIAL-LICENSING.md
- Required notice: https://github.com/arjunmaybe/Searchiee/blob/main/NOTICE
- Issues: https://github.com/arjunmaybe/Searchiee/issues

Product facts used on the site (downloadable v0.2.0):

- Searchiee is a Windows 10/11 x64 desktop app, not a hosted service.
- It indexes only Markdown (`.md`) and plain-text (`.txt`) files.
- Users add and refresh folders manually; there is no background filesystem
  watcher.
- Embeddings are generated locally with FastEmbed `AllMiniLML6V2`
  (384 dimensions).
- Long documents are split into overlapping chunks sized with the model's
  tokenizer, with a 4,096-character secondary cap and up to 120 characters of
  context overlap when the token budget allows. Up to 32 chunk texts are
  embedded per model call.
- The local index stores registered folder paths, document paths, contents,
  and per-chunk embeddings in `index-v1.json` inside Tauri's application-specific
  local data directory (current schema v3). Legacy whole-file entries remain
  supported.
- Search returns one result per file, ranking each file by its best-matching
  chunk. Scores below `0.25` are filtered out; the top 10 results show a
  snippet of up to 200 characters from the best-matching chunk.
- The `0.25` minimum cosine-similarity threshold is experimental and is not
  calibrated against a broad labelled evaluation set.
- The first model download is roughly 90 MB and requires network access.
  Later indexing and search work offline once the model is cached.
- v0.2.0 contains no telemetry, account system, or cloud search service.
- Release v0.2.0 was published on 21 November 2026. The release includes
  Windows NSIS and MSI installers and a SHA-256 manifest.

## Licensing facts used on the site

- Searchiee is source-available under the PolyForm Noncommercial License
  1.0.0. It is not OSI-approved open-source software.
- The license permits noncommercial purposes and expressly covers personal
  uses and the listed qualifying organizations, as defined by its terms.
- General commercial use—including internal business operations, commercial
  redistribution, resale, bundling, and paid hosted services—requires separate
  written permission unless the community license expressly covers the use.
- The commercial licensing document's 5% attributable-net-receipts royalty
  and annual minimum guarantee are proposed negotiating terms, not automatic
  obligations or a binding public offer. Rights and payment obligations arise
  only under a separate written agreement.
- Redistributors must provide the license text or its official URL and retain
  the required notice: `Required Notice: Copyright (c) 2026 Arjun Sasi.`
- Third-party dependencies and the embedding model remain subject to their
  own licenses and terms.

## Missing product image

There is no suitable high-resolution application screenshot in the
repository. The existing app icon (`src-tauri/icons/icon.ico`) is a multi-size
ICO containing 16, 24, 32, 48, 64, 128, and 256 pixel frames. The site uses the
same file as its favicon, linked at 32×32. It is not a product screenshot. Do
not substitute a fabricated interface or a broken placeholder.

The exact screenshot to capture later:

1. Run Searchiee 0.2.0 from an isolated synthetic demo library. Do not use a
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

The custom production domain is intentionally deferred. Cloudflare Pages
provides the automatic `pages.dev` deployment URL, but do not attach a custom
domain or treat the current deployment as launch-complete while these tasks
are unresolved. Do not use localhost, a preview URL, or a guessed domain as a
canonical.

Before launch, with the verified production origin:

- Add a self-referencing canonical URL to every indexable page.
- Add `og:url` and an absolute `og:image` URL, and switch the Twitter card to
  `summary_large_image` if a real social image is ready.
- Add `robots.txt` and `sitemap.xml` with absolute production URLs.
- Verify HTTPS, canonical links, Open Graph previews, and the 404 status after
  deployment.
- Recheck the installer, checksums, release, and issue URLs at launch.

The current HTML intentionally omits canonical, `og:url`, `og:image`, and
sitemap files. It does not contain fake placeholders.

## Legal and factual details still requiring the owner

- Keep the website license summary aligned with `LICENSE` and `COMMERCIAL-LICENSING.md`; do not describe Searchiee as OSI-approved open source.
- Decide whether a dedicated privacy contact email and legal entity name are
  required for the jurisdictions where the site and app are offered.
- Confirm whether a separate terms page or installer EULA is needed.
- Approve the privacy wording for Cloudflare Pages hosting, because hosting
  request logs are controlled by that provider.

The privacy page currently identifies GitHub Issues as the verified contact
channel and states that a privacy email or legal entity is not published.

## Deployment

The site deploys to Cloudflare Pages through GitHub Actions and Wrangler.

- Workflow: `.github/workflows/deploy-searchiee-pages.yml`
- Cloudflare Pages project: `searchiee`
- Production URL: `https://searchiee.pages.dev` (HTTP 200 verified 7 November 2026)
- Production branch: `main`
- Preview branch: `feat/searchiee-website`

When a push changes `website/**` or the deployment workflow:

- A push to `feat/searchiee-website` creates a preview deployment for
  `feat/searchiee-website`.
- A push to `main` creates the production deployment for `main`.
- Manual workflow runs deploy the branch selected in GitHub Actions.
- Pull-request events do not deploy.

The deployment branch is supplied dynamically with `${{ github.ref_name }}`.
Manual runs must select a branch; tag refs are rejected before deployment.
Each branch uses its own concurrency group, so preview deployments cannot
cancel production deployments.

The workflow uses Node.js 22, runs `npm ci` and `npm run build` from
`website/`, and uploads `website/dist/` with
`cloudflare/wrangler-action@v4`. It preserves the `searchiee` project name and
uses these GitHub repository secrets:

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`

`GITHUB_TOKEN` is provided automatically by GitHub Actions.

Cloudflare Pages automatically adds `X-Robots-Tag: noindex` to preview
deployment responses. Verify a representative preview and confirm the header:

```powershell
curl -I "https://<preview-deployment>.searchiee.pages.dev"
```

Replace the placeholder with the preview URL reported by the deployment. The
response should include:

```text
x-robots-tag: noindex
```

This automatic preview protection does not replace the custom-domain,
canonical URL, sitemap, robots, or social metadata work.

The build remains a portable static site:

- Build command: `npm ci && npm run build` from `website/`
- Publish directory: `website/dist`
- Node.js: 20 or newer; CI uses 22
- Verify Cloudflare Pages returns `404.html` with an HTTP 404 status for
  unknown paths.
- Set security headers at the host if supported. Keep the site free of
  third-party runtime requests.
- Do not add analytics without a concrete measurement need and an accurate
  privacy disclosure.

## Scope safeguards

- Website source and configuration belong under `website/`. The Cloudflare
  Pages workflow is the only deployment workflow outside that directory.
- Do not change the root Searchiee package, lockfile, Tauri configuration,
  desktop behavior, indexing, persistence, ranking, or release functionality.
  Keep public product claims aligned with the downloadable v0.2.0 release.
- Do not touch `clippy-out.txt`, evaluation outputs, `test-out.txt`, or
  `src-tauri/.fastembed_cache/`.
- Do not add unrelated CI workflows or change the Cloudflare Pages deployment
  workflow without separate review and approval.
