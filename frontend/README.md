# Enterprise Document Intelligence — Frontend

React + Vite + Tailwind CSS UI for the FastAPI backend.

## Stack
- **React 18** + **Vite 6** (fast dev server / build)
- **Tailwind CSS 3** (utility-first, with a small component layer in `index.css`)
- No state library — local component state + a thin `fetch` client (`src/api.js`)

## Setup

```bash
npm install
cp .env.example .env        # point VITE_API_URL at the backend (default :8000)
npm run dev                 # http://localhost:5173
```

Make sure the backend is running (`uvicorn app.api.main:app --port 8000`).

## Build

```bash
npm run build && npm run preview
```

## Features

- **Authentication** — sign up / sign in gate the app. Client-side (localStorage)
  with **SHA-256 password hashing** (never plaintext) via `AuthContext`. Swap the
  `signup`/`signin` calls for a real API to back it with a server + JWT.
- **Google OAuth** — "Continue with Google" via Google Identity Services.
  Set `VITE_GOOGLE_CLIENT_ID` (an OAuth 2.0 Web Client ID with
  `http://localhost:5173` as an authorized origin) to enable the real flow;
  without it, a clearly-labelled **demo** fallback signs in a sample Google
  account so the button is functional out of the box.
- **Dark / light theme** — `ThemeContext` persists the choice and respects the OS
  `prefers-color-scheme`; an inline script in `index.html` applies it before
  first paint (no flash). Toggled from the header. Tailwind `darkMode: "class"`
  drives `dark:` variants throughout.
  > ⚠️ If the toggle seems inert, **restart `npm run dev`** — `darkMode` is read
  > from `tailwind.config.js` at server start, so a config change made while the
  > dev server was running won't take effect until restart.
- **Four workspaces** — Analyze, Upload, Search, Graph (see below).

## Layout

```
src/
  api.js                   fetch client for the 6 endpoints
  App.jsx                  shell: auth gate, header, tabs, theme + user menu
  index.css                Tailwind + reusable themed classes (.card/.btn/.chip…)
  context/
    AuthContext.jsx        localStorage auth, SHA-256 hashing, session
    ThemeContext.jsx       dark/light theme + OS preference + persistence
  lib/labels.js            per-entity-label color theme
  components/
    AuthScreen.jsx         sign in / sign up forms + Google OAuth
    GoogleButton.jsx       Google Identity Services button (+ demo fallback)
    ThemeToggle.jsx        sun/moon theme switch
    UserMenu.jsx           avatar (or Google photo) + sign-out dropdown
    AnalyzePanel.jsx       text → agent workflow, inline entity highlighting
    UploadPanel.jsx        drag-drop document upload + extraction
    SearchPanel.jsx        semantic search over indexed docs
    GraphPanel.jsx         knowledge-graph triple-pattern query + stats
    HighlightedText.jsx    renders text with color-coded entity spans
    EntityList.jsx         entity chips
    RelationList.jsx       relation triples
    Section.jsx            labeled section wrapper
```

## Design notes
- **Industry-standard Tailwind**: utilities in markup, a thin `@layer components`
  for repeated patterns (`.card`, `.btn-primary`, `.input`, `.chip`), Inter +
  JetBrains Mono via Google Fonts, a slate/indigo palette, custom scrollbars, and
  a subtle `fade-in` animation.
- **Entity highlighting** is offset-based: the backend returns character spans,
  the UI slices the source text and wraps each span in a colored `<mark>` — the
  same span data that powers BIO tagging now drives the visualization.
- **Accessible & responsive**: keyboard-submittable inputs, drag-drop with
  visible state, responsive grids (`lg:grid-cols-2`, `sm:grid-cols-3`).
