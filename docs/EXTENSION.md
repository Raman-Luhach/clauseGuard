# ClauseGuard browser extension

The extension brings ClauseGuard to any web page: open a terms of service, a privacy policy or
a contract, click the ClauseGuard button, and the flagged clauses are highlighted on the page
itself, with plain-language explanations on hover. It works in Chrome, Edge, Brave, Arc and
other Chromium browsers (Manifest V3, Chrome 116 or later).

The extension does no analysis itself: it sends the page's text to a ClauseGuard server (the
same app as the web UI) and shows the answer. Use your own computer (`make serve`) while
testing, and the Hugging Face Space once it is deployed (`docs/DEPLOY.md`).

## What it does

- **Toolbar button → popup.** Reads the page (your selection if you selected at least 50 words,
  else the page's main content, else the whole page), asks the server what kind of document
  it is and pre-selects that type (you can change it), then analyses it on **Analyse this
  page**. The popup shows the risk score and the most severe flags; click a flag to scroll to it.
- **Highlights on the page,** coloured by severity. Hover over a highlight for its category and
  explanation. **Remove highlights** restores the page.
- **Full report** in a tab: every flag with the clause text, explanation, risk group and
  confidence, plus the protections found or missing in a privacy policy.
- **Right-click → "Analyse selected text with ClauseGuard"** on any selection; the type is
  detected automatically and the report opens in a tab.
- **Settings:** the server address, with a connection test.

Pages the browser protects (the Chrome Web Store, `chrome://` pages) can't be read. PDFs
opened in the browser's PDF viewer can't be read either: upload them in the web app instead.

## Try it (5 minutes)

1. Start the server on the Mac: `cd ~/nlp/clauseguard && make serve` (http://127.0.0.1:8000).
2. Open `chrome://extensions`, switch on **Developer mode** (top right), click **Load unpacked**
   and choose the `extension/` folder of this repository.
3. Pin ClauseGuard (puzzle-piece icon → pin). Its settings point to `http://127.0.0.1:8000`
   by default.
4. Open any terms of service page (for example https://docs.github.com/en/site-policy/github-terms/github-terms-of-service),
   click the ClauseGuard button, check the suggested type and press **Analyse this page**.

After changing extension files, press the reload icon on its card in `chrome://extensions`.

## Use the deployed server

After deploying the Space (`docs/DEPLOY.md`), open the extension's **Settings** and enter the
Space's direct address, e.g. `https://<your-user>-clauseguard.hf.space`, then **Test
connection**. The server allows requests from browser extensions (CORS for
`chrome-extension://` and `moz-extension://` origins); allow other web origins with
`CLAUSEGUARD_CORS_ORIGINS` if you ever need to.

A free Space sleeps after a while without visitors; the first analysis afterwards waits for it
to start (about a minute). Analysing a 5,000-word document takes roughly 10–20 seconds on the
free CPU.

## Publish it on the Chrome Web Store

1. `make extension` writes `dist/clauseguard-extension.zip`.
2. Register as a Chrome Web Store developer at
   https://chrome.google.com/webstore/devconsole (one-time $5 fee).
3. **New item** → upload the zip. Fill in the listing: description, category
   (Productivity), at least one 1280×800 screenshot of the popup and of highlights on a page,
   and the 128×128 icon (`extension/icons/icon128.png`).
4. **Privacy practices:** declare that the extension handles "website content" only to provide
   its single purpose, sends it to the ClauseGuard server, and does not sell or use it for
   anything else. A privacy policy URL is required; the text below can be published as a page
   (for example in the Space's README or a GitHub page).
5. Before publishing, change the default server in `extension/lib/core.js`
   (`DEFAULT_API`) to the Space's address, so users don't have to set it.
6. Submit for review (usually a few days).

**Licence note:** the privacy model is trained on OPP-115, which allows non-commercial use only.
Publish the extension free of charge and say so in the listing.

### Privacy policy (template)

> ClauseGuard reads the text of a web page only when you click its button or choose "Analyse
> selected text with ClauseGuard". That text is sent over HTTPS to the ClauseGuard server
> configured in the extension's settings, which analyses it in memory and does not store it;
> the server logs only counts and timings (number of words, number of flags, processing
> time), never the text. The extension stores only its settings and your last report, in your
> browser. No data is sold, shared with third parties or used for advertising. If the server
> is configured to use an AI model for explanations, only the individual flagged clauses
> (never the whole document) are sent to that model's provider.

## Permissions

| Permission | Why |
|---|---|
| `activeTab` | read the page you are on, only after you click the button |
| `scripting` | insert the highlighting script into that page |
| `storage` | the server address (synced) and your last report (this browser session only) |
| `contextMenus` | the "Analyse selected text" item |

No host permissions and no access to other tabs: the extension can't read pages in the
background.

## Files

| File | Purpose |
|---|---|
| `manifest.json` | extension manifest (MV3) |
| `popup.html/js` | toolbar popup: detect type, analyse, summary |
| `content.js` | injected into the page on click: reads text, highlights clauses |
| `report.html/js` | full report tab |
| `options.html/js` | server address |
| `background.js` | right-click menu |
| `lib/core.js` | shared helpers (text matching, API calls); tested in `tests/js/extension.test.mjs` |
| `ui.css`, `icons/` | styles (light and dark) and icons |
