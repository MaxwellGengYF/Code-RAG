# Unity Manual & Script API Reference (offline mirror)

Downloaded from https://docs.unity3d.com (current docs, Unity 6.x era) on 2026-08-28.

## Contents

- `Manual/` — Unity Manual (HTML). Official TOC has 3,550 pages; 3,546 saved.
- `ScriptReference/` — Unity Scripting API Reference (HTML). Official TOC has 4,774
  pages; 4,773 saved, plus ~40k additional API member pages discovered via links.
- `StaticFilesManual/`, `StaticFilesScriptReference/`, `uploads/` — CSS, JS, images,
  fonts and other assets referenced by the pages (same-host only).

Total: ~46,200 files, ~1.03 GB.

## Notes

- Open `Manual/index.html` or `ScriptReference/index.html` in a browser to browse.
  Relative links between pages work offline because the URL structure is mirrored.
- A few TOC pages are dead links on Unity's server (redirect loops / 404s) and were
  not saved; they are listed in `.crawler_state.json` as `missing`.
- External links (cdn.cookielaw.org, developer.meta.com, unity.com, other-language
  mirrors `/cn/`, `/ja/`, `/kr/`, etc.) were intentionally not downloaded.
- `.crawler_state.json` is the crawler's resume state (which URLs are done/missing);
  it can be deleted if not needed.

## How it was downloaded

`D:/unity_docs_downloader.py` — resumable, polite concurrent crawler (8 workers,
random 0.15–0.35 s delay, retries). Seed list from `Manual/docdata/toc.js` and
`ScriptReference/docdata/toc.js`; assets discovered by parsing HTML/CSS references.
