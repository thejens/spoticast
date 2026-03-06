"""FastAPI application — routes, background jobs, SSE progress streaming."""

from __future__ import annotations

import asyncio
import json
import logging
from uuid_extensions import uuid7

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from spoticast.config import settings
from spoticast import episodes as episodes_store
from spoticast.api import audio as audio_api
from spoticast.api import gemini as claude_api
from spoticast.api import lastfm as lastfm_api
from spoticast.api import research as research_api
from spoticast.api import spotify as spotify_api
from spoticast.api import tts as tts_api

app = FastAPI(title="Spoticast")

# Serve generated audio files
_GENERATED_DIR = Path("generated")
_GENERATED_DIR.mkdir(exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(_GENERATED_DIR)), name="audio")

# Serve frontend static files
_WEB_DIR = Path(__file__).parent / "web"
app.mount("/web", StaticFiles(directory=str(_WEB_DIR)), name="web")


# ---------------------------------------------------------------------------
# In-memory job state
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, job_id: str, playlist_uri: str):
        self.job_id = job_id
        self.playlist_uri = playlist_uri
        self.playlist_name: str = ""
        self.track_uris: list[str] | None = None
        self.status: str = "pending"   # pending | running | done | error
        self.events: list[dict] = []
        self.queue: list[dict] | None = None
        self.error: str | None = None
        self._event = asyncio.Event()

    def push(self, event_type: str, data: Any):
        self.events.append({"type": event_type, "data": data})
        self._event.set()
        self._event.clear()

    def push_threadsafe(self, loop: asyncio.AbstractEventLoop, event_type: str, data: Any):
        """Thread-safe variant for use from run_in_executor threads."""
        self.events.append({"type": event_type, "data": data})
        loop.call_soon_threadsafe(self._event.set)

    async def wait_for_event(self):
        await self._event.wait()


_jobs: dict[str, Job] = {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    html_path = _WEB_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)


@app.get("/auth/spotify")
async def auth_spotify():
    url = spotify_api.get_auth_url()
    return RedirectResponse(url)


@app.get("/auth/callback")
async def auth_callback(code: str | None = None, error: str | None = None):
    if error or not code:
        return HTMLResponse(
            f"<h1>Auth failed</h1><p>{error}</p>", status_code=400
        )
    spotify_api.handle_callback(code)
    return RedirectResponse("/?auth=success")


@app.get("/auth/token")
async def auth_token():
    token = spotify_api.get_current_token()
    if token is None:
        return JSONResponse({"token": None, "authenticated": False})
    return JSONResponse({"token": token, "authenticated": True})


@app.get("/auth/lastfm/status")
async def lastfm_status():
    return JSONResponse(lastfm_api.get_status())


class LastFMConnectRequest(BaseModel):
    username: str


@app.post("/auth/lastfm/connect")
async def lastfm_connect(req: LastFMConnectRequest):
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, lastfm_api.connect, req.username.strip())
        return JSONResponse(result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/auth/lastfm/disconnect")
async def lastfm_disconnect():
    lastfm_api.disconnect()
    return JSONResponse({"connected": False})


@app.get("/api/recent")
async def api_recent():
    token = spotify_api.get_current_token()
    if token is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    loop = asyncio.get_event_loop()
    playlists = await loop.run_in_executor(None, spotify_api.fetch_recent_plays)
    return JSONResponse({"playlists": playlists})



@app.get("/api/playlists")
async def api_playlists(limit: int = 50, offset: int = 0):
    token = spotify_api.get_current_token()
    if token is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: spotify_api.fetch_user_playlists(limit, offset)
    )
    return JSONResponse(result)


@app.get("/api/episodes")
async def api_episodes():
    return JSONResponse({"episodes": episodes_store.list_episodes()})


@app.get("/api/episodes/{episode_id}")
async def api_episode(episode_id: str):
    ep = episodes_store.get_episode(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return JSONResponse(ep)


class GenerateRequest(BaseModel):
    playlist_uri: str | None = None
    track_uris: list[str] | None = None


@app.post("/generate")
async def generate(req: GenerateRequest):
    token = spotify_api.get_current_token()
    if token is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not req.playlist_uri and not req.track_uris:
        raise HTTPException(status_code=400, detail="Provide playlist_uri or track_uris")

    job_id = str(uuid7())
    source = req.playlist_uri or "track_list"
    job = Job(job_id, source)
    job.track_uris = req.track_uris
    _jobs[job_id] = job

    # Run generation in background so we can stream progress
    asyncio.create_task(_run_generation(job))

    return {"job_id": job_id}


@app.get("/jobs/{job_id}/stream")
async def job_stream(job_id: str, request: Request):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        sent_index = 0
        while True:
            if await request.is_disconnected():
                break

            # Drain any new events
            while sent_index < len(job.events):
                ev = job.events[sent_index]
                sent_index += 1
                yield {"event": ev["type"], "data": json.dumps(ev["data"])}

            if job.status in ("done", "error"):
                break

            # Wait for new events (with timeout to allow disconnect checks)
            try:
                await asyncio.wait_for(job.wait_for_event(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    return EventSourceResponse(event_generator())


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    path = _GENERATED_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(str(path), media_type="audio/mpeg")


# ---------------------------------------------------------------------------
# Generation pipeline
# ---------------------------------------------------------------------------

async def _run_generation(job: Job):
    """Background task: fetch data → generate script → synthesize audio → save episode."""
    loop = asyncio.get_event_loop()
    episode_id = job.job_id

    try:
        job.status = "running"
        job.push("progress", {"step": "fetch", "message": "Fetching playlist and user data..."})

        # Fetch tracks — either from a playlist URI or from a direct track list
        if job.track_uris:
            tracks = await loop.run_in_executor(
                None, spotify_api.fetch_tracks, job.track_uris[:settings.max_tracks]
            )
            job.playlist_name = "Custom tracks"
        else:
            tracks, job.playlist_name = await asyncio.gather(
                loop.run_in_executor(None, spotify_api.fetch_playlist, job.playlist_uri),
                loop.run_in_executor(None, spotify_api.fetch_playlist_name, job.playlist_uri),
            )
            tracks = tracks[:settings.max_tracks]

        if not tracks:
            raise ValueError("No tracks found. Check the playlist link or track list.")

        job.push("progress", {"step": "fetch", "message": f"Fetched {len(tracks)} tracks. Loading audio features..."})

        track_uris = [t.uri for t in tracks]
        features, user_ctx = await asyncio.gather(
            loop.run_in_executor(None, spotify_api.fetch_audio_features, track_uris),
            loop.run_in_executor(None, spotify_api.fetch_user_context),
        )

        job.push("progress", {"step": "context", "message": "Building listener profile..."})

        context = spotify_api.build_playlist_context(tracks, features, user_ctx)

        # Last.fm enrichment: fan-era, play counts, loved tracks, artist bios/similar
        if lastfm_api.is_configured():
            job.push("progress", {"step": "research", "message": "Fetching Last.fm history and artist data..."})

            def _lastfm_progress(message: str):
                job.push_threadsafe(loop, "progress", {"step": "research", "message": message})

            context = await loop.run_in_executor(
                None, lambda: lastfm_api.enrich_context(context, progress_cb=_lastfm_progress)
            )

        # Grounded research: Gemini + Google Search for interviews, news, stories
        job.push("progress", {"step": "research", "message": "Researching artists and songs..."})

        def _research_progress(message: str):
            job.push_threadsafe(loop, "progress", {"step": "research", "message": message})

        context = await loop.run_in_executor(
            None, lambda: research_api.enrich_with_research(context, progress_cb=_research_progress)
        )

        job.push("progress", {"step": "script", "message": "Generating podcast script with Gemini..."})

        # Attach playlist name to context so generate_episode_name can use it
        context["playlist_name"] = job.playlist_name

        script, episode_name = await asyncio.gather(
            claude_api.generate_script(context),
            claude_api.generate_episode_name(context),
        )

        job.push("progress", {"step": "tts", "message": "Synthesizing intro audio..."})

        ep_dir = episodes_store.episode_audio_dir(episode_id)
        saved_queue: list[dict] = []

        # --- Synthesize intro, then start streaming immediately ---
        intro_file = f"{ep_dir}/intro.mp3"
        intro_pcm = await tts_api.synthesize_dialogue(script["intro"])
        await loop.run_in_executor(
            None, audio_api.assemble_commentary, intro_pcm, intro_file
        )
        saved_queue.append({"type": "audio", "url": f"/audio/{intro_file}"})
        job.push("intro_ready", {"url": f"/audio/{intro_file}", "episode_name": episode_name})

        # --- Synthesize per-track commentary, streaming each as it finishes ---
        total_tracks = len(script["tracks"])
        for i, track_script in enumerate(script["tracks"]):
            job.push("progress", {
                "step": "tts",
                "message": f"Synthesizing commentary for track {i+1}/{total_tracks}...",
            })

            commentary_file = f"{ep_dir}/track_{i:03d}_commentary.mp3"
            pcm = await tts_api.synthesize_dialogue(track_script["commentary"])
            await loop.run_in_executor(
                None, audio_api.assemble_commentary, pcm, commentary_file
            )

            saved_queue.append({"type": "audio", "url": f"/audio/{commentary_file}"})
            saved_queue.append({"type": "spotify", "uri": track_script["track_uri"]})

            job.push("track_ready", {
                "index": i,
                "total": total_tracks,
                "commentary_url": f"/audio/{commentary_file}",
                "track_uri": track_script["track_uri"],
            })

        # --- Synthesize outro ---
        if script.get("outro"):
            job.push("progress", {"step": "tts", "message": "Synthesizing outro..."})
            outro_pcm = await tts_api.synthesize_dialogue(script["outro"])
            await loop.run_in_executor(
                None, audio_api.assemble_commentary, outro_pcm, "outro.mp3"
            )
            saved_queue.append({"type": "audio", "url": "/audio/outro.mp3"})
            job.push("outro_ready", {"url": "/audio/outro.mp3"})

        # Persist episode for future playback
        episodes_store.save_episode(
            episode_id=episode_id,
            name=episode_name,
            playlist_uri=job.playlist_uri,
            playlist_name=job.playlist_name,
            track_count=total_tracks,
            queue=saved_queue,
        )

        job.status = "done"
        job.push("done", {"episode_id": episode_id, "episode_name": episode_name})

    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        # Produce a friendlier message for common Spotify API errors
        msg = str(exc)
        if "404" in msg:
            msg = "Playlist not found. Make sure it's public (or that you own it) and that the URI is correct."
        elif "401" in msg or "403" in msg:
            msg = "Spotify auth error. Try disconnecting and reconnecting your Spotify account."
        job.push("error", {"message": msg})
        logger.exception("Generation failed for job %s", job.job_id)
