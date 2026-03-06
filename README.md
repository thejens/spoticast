# Spoticast

Your Spotify playlist, reimagined as a podcast. Two AI hosts deliver DJ intros and per-track commentary, interleaved with the actual music — all in the browser.

## How it works

1. Paste a Spotify playlist link
2. Spoticast fetches the tracks, audio features, artist bios, and (optionally) your Last.fm listening history
3. Gemini generates a podcast script: a ~5-minute intro + per-track commentary with trivia, production stories, and artist connections
4. Gemini TTS synthesizes two voices in one pass (multi-speaker)
5. The browser player crossfades between commentary audio and Spotify playback via the Web Playback SDK

Audio is streamed to the player as soon as the intro is synthesized — you don't wait for the whole episode.

## Prerequisites

- **Spotify Premium** — required for the Web Playback SDK (in-browser playback)
- **ffmpeg** — required by pydub: `brew install ffmpeg`
- **Python 3.13+**
- **`uv`** — `brew install uv`

## Setup

### 1. Spotify app

Go to [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard), create an app, and add this redirect URI under **Edit Settings**:

```
http://127.0.0.1:8765/auth/callback
```

Copy the **Client ID** and **Client Secret** from the app dashboard.

> The Client Secret is technically optional (Spoticast uses PKCE for the browser OAuth flow) but recommended for server-side token refresh.

### 2. Gemini access (choose one)

**Option A — Google AI Studio API key** (easiest, no GCP account needed):

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Create an API key
3. Set `GEMINI_API_KEY=...` in your `.env`

**Option B — Vertex AI via Google Cloud** (for GCP users):

```bash
gcloud auth application-default login
```

Set `GOOGLE_CLOUD_PROJECT=your-project-id` in `.env`, or let it resolve from ADC automatically.

> Vertex AI requires the **Vertex AI API** to be enabled in your GCP project and billing to be set up.

### 3. Last.fm (optional)

Last.fm enriches the commentary with your play counts, loved tracks, and listening era context. Without it, Spoticast still works — it just won't reference your personal history.

1. Create an API account at [last.fm/api/account/create](https://www.last.fm/api/account/create)
2. Copy the **API Key** and **Shared Secret** into `.env`

### 4. Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` and fill in your keys. A minimal setup (AI Studio, no Last.fm) looks like this:

```env
SPOTIFY_CLIENT_ID=abc123...
SPOTIFY_CLIENT_SECRET=def456...
GEMINI_API_KEY=AIza...
```

A full setup with Last.fm and Vertex AI:

```env
SPOTIFY_CLIENT_ID=abc123...
SPOTIFY_CLIENT_SECRET=def456...

# Leave GEMINI_API_KEY unset to use Vertex AI via ADC instead
GOOGLE_CLOUD_PROJECT=my-gcp-project

LASTFM_API_KEY=abc123...
LASTFM_API_SECRET=def456...
```

> The Last.fm widget in the UI is only shown when `LASTFM_API_KEY` and `LASTFM_API_SECRET` are present — you can safely omit them and the UI stays clean.

### 5. Install and run

```bash
uv sync
make run
```

The browser opens automatically at `http://127.0.0.1:8765`.

### 5. In the browser

1. Click **Connect Spotify** and complete the OAuth flow
2. Optionally enter your Last.fm username and click **Connect**
3. Paste a playlist link, URI, or select one from your library
4. Click **Generate Podcast** — the intro plays as soon as it's ready (~1–2 min), remaining tracks stream in as they're synthesized

## Spotify API limitations (development mode)

Spotify apps start in **Development Mode**, which has a few practical constraints:

**Only your own account can connect.** To let other people use your instance, you must add them explicitly under *Users and Access* in the Spotify developer dashboard (max 25 users). Without this, anyone else who tries to connect will get an auth error.

**Playlist access is restricted to playlists you own.** You can't directly paste a link to someone else's playlist (e.g. Spotify editorial playlists, a friend's playlist) — Spotify will return a 404. Two workarounds:

- **Best option — copy-paste tracks:** Open any playlist in the Spotify desktop app, press `⌘A` (Mac) / `Ctrl+A` (Windows) to select all tracks, then `⌘C` / `Ctrl+C` to copy. Paste directly into the Spoticast input box. This works for any playlist including Discover Weekly, Release Radar, editorial playlists, etc.
- **Save a copy:** Use Spotify's *Add to playlist* to copy the tracks into a playlist you own, then paste that playlist's link.

To remove the 25-user cap and playlist restrictions, you can apply for a [Spotify Extended Quota](https://developer.spotify.com/documentation/web-api/concepts/quota-modes) — but for personal use the copy-paste workaround is simpler.

## Configuration reference

All settings are read from `.env` (or environment variables):

| Variable | Required | Description |
|---|---|---|
| `SPOTIFY_CLIENT_ID` | Yes | From the Spotify developer dashboard |
| `SPOTIFY_CLIENT_SECRET` | Recommended | Enables server-side token refresh |
| `GEMINI_API_KEY` | One of these | Google AI Studio API key |
| `GOOGLE_CLOUD_PROJECT` | ↑ or ADC | GCP project for Vertex AI |
| `LASTFM_API_KEY` | No | Enriches commentary with listening history |
| `LASTFM_API_SECRET` | No | Required alongside `LASTFM_API_KEY` |
| `GEMINI_MODEL` | No | Defaults to `gemini-2.5-pro` |
| `MAX_TRACKS` | No | Tracks per episode, default `30` |
| `PORT` | No | Default `8765` |

## Makefile

```
make run          # Start the app
make dev          # Start with auto-reload
make clean-cache  # Clear generated audio and script cache (forces regeneration)
make clean        # Remove venv and Python build artifacts
```

## Architecture

```
spoticast/
  __main__.py      # Entry point: starts uvicorn + opens browser
  config.py        # Pydantic settings (from .env / env vars)
  cache.py         # Disk-backed JSON cache for scripts and research
  server.py        # FastAPI: routes, background jobs, SSE progress stream
  api/
    spotify.py     # OAuth, playlist fetch, audio features, user context
    gemini.py      # Script generation (Gemini 2.5 Pro, structured JSON output)
    tts.py         # Multi-speaker TTS (Gemini 2.5 Flash TTS)
    audio.py       # pydub: PCM → MP3 assembly
    lastfm.py      # Fan-era classification, artist profiles, track enrichment
    research.py    # Wikipedia parallel fetch with disk cache
  web/
    index.html     # Single-page UI
    player.js      # Spotify SDK + HTML Audio interleaved queue + crossfade
    styles.css
  generated/       # Runtime MP3 files (gitignored)
```
