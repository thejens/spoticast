"""
Web research enrichment — Gemini + Google Search grounding for artists and songs.

Uses Vertex AI's grounded generation to find interviews, news articles,
recording stories, reviews, and cultural context for each artist and track.

Features:
  - Google Search grounding: Gemini searches the web for real, current information
  - Parallel batching: artists and songs researched concurrently
  - Persistent disk cache in .research_cache/research/
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import google.auth
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel, Tool
from vertexai.generative_models._generative_models import gapic_tool_types as _gapic

from spoticast import cache
from spoticast.config import settings

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "research"
_MAX_WORKERS = 4
_initialized = False


def _ensure_init():
    global _initialized
    if _initialized:
        return
    project = settings.google_cloud_project
    if not project:
        _, project = google.auth.default()
    vertexai.init(project=project, location=settings.google_cloud_location)
    _initialized = True


def _get_model() -> GenerativeModel:
    _ensure_init()
    return GenerativeModel(settings.gemini_research_model)


# Use google_search (not google_search_retrieval) — required by newer Gemini models on Vertex AI
_SEARCH_TOOL = Tool._from_gapic(raw_tool=_gapic.Tool(google_search=_gapic.Tool.GoogleSearch()))


# ─────────────────────────────────────────────────────────────────────────────
# Grounded research via Gemini
# ─────────────────────────────────────────────────────────────────────────────

def _grounded_query(prompt: str, max_output_tokens: int = 1500) -> str:
    """Run a single grounded Gemini query and return the text response."""
    model = _get_model()
    response = model.generate_content(
        prompt,
        tools=[_SEARCH_TOOL],
        generation_config=GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=0.3,
        ),
    )
    return response.text.strip() if response.text else ""


# Cache artist origin lookups separately — they're reused across song research
_origin_cache: dict[str, str] = {}


def _get_artist_origin(artist_name: str) -> str:
    """
    Quick grounded lookup to establish an artist's country and native language.

    Returns a short string like "Swedish (Sverige)" or "French (France)" that
    gets injected into subsequent search prompts so the model issues queries in
    the right language. Cached in-memory for the process lifetime.
    """
    if artist_name in _origin_cache:
        return _origin_cache[artist_name]

    try:
        origin = _grounded_query(
            f'What country is the musical artist "{artist_name}" from, and what is their native language? '
            f"Reply in exactly this format (one line): Country: X | Language: X | Native name: X "
            f"(native name = how the artist's name or their country is written in the native script/language, "
            f"e.g. 'Sverige' for Sweden, '日本' for Japan). If English-speaking, reply: Country: X | Language: English | Native name: -",
            max_output_tokens=80,
        )
    except Exception:
        origin = ""

    _origin_cache[artist_name] = origin
    return origin


def _research_artist(artist_name: str) -> dict:
    """Research an artist using Gemini + Google Search. Returns cached result if available."""
    key = cache.cache_key("artist_v3", artist_name)
    cached = cache.get(_CACHE_PREFIX, key)
    if cached is not None:
        return cached

    origin = _get_artist_origin(artist_name)
    native_lang_instruction = (
        f"The artist's origin: {origin}\n"
        f"Issue searches in BOTH English AND the native language shown above — you need both.\n"
        f"English searches: look for interviews in Rolling Stone, Pitchfork, NME, The Guardian, NPR Music, "
        f"AllMusic, and international press. Search e.g. '\"{artist_name}\" interview', '\"{artist_name}\" biography'.\n"
        f"Native-language searches: search using the artist's name plus native-language keywords "
        f"(e.g. interview, biographie, インタビュー, karriere, musik, биография — whatever fits). "
        f"Check native-language Wikipedia, local music press, and fan sites for detail absent from English coverage."
        if origin else
        f"Search in English across major music press (Rolling Stone, Pitchfork, NME, The Guardian, AllMusic). "
        f"If you find evidence the artist is non-English-speaking, also search in their native language."
    )

    prompt = f"""Research the musical artist "{artist_name}" for a podcast about their music.

{native_lang_instruction}

Find and summarize in 3-5 concise paragraphs:
- Origin and background: where they're from, their cultural/musical roots, what scene they emerged from
- Who they are: genre, career arc, what makes them distinctive
- Notable interviews or artist quotes about their creative process or philosophy (translate non-English quotes)
- Recent news, tours, album releases, or notable events (last 2 years)
- Critical reception, awards, or cultural impact — including in their home country
- Interesting stories: recording sessions, collaborations, influences, controversies

Focus on specific facts, quotes, and stories that would make great podcast talking points.
Do NOT pad with generic statements. If you can't find much, keep it short."""

    result: dict = {"name": artist_name, "origin": origin, "research": ""}
    try:
        text = _grounded_query(prompt)
        if text:
            result["research"] = text
    except Exception as exc:
        logger.warning("Grounded research failed for artist %s: %s", artist_name, exc)

    # Only cache successful results — empty/failed results should be retried next run
    if result["research"]:
        cache.put(_CACHE_PREFIX, key, result)
    return result


def _research_song(artist_name: str, song_title: str, spotify_track_id: str) -> dict:
    """Research a song using Gemini + Google Search. Keyed by Spotify track ID."""
    key = cache.cache_key("song_v3", spotify_track_id)
    cached = cache.get(_CACHE_PREFIX, key)
    if cached is not None:
        return cached

    primary = artist_name.split(",")[0].strip()

    # Reuse origin from artist research if already fetched (likely warm from parallel artist pass)
    origin = _get_artist_origin(primary)
    native_lang_instruction = (
        f"The artist's origin: {origin}\n"
        f"Issue searches for this song in BOTH English AND the artist's native language — you need both.\n"
        f"English searches: look in Pitchfork, Rolling Stone, NME, Genius, AllMusic, and international press "
        f"for reviews, interviews, and lyrics analysis.\n"
        f"Native-language searches: search the song title in the native language, check local lyrics/music sites, "
        f"local press reviews, and any interviews where the artist discussed this track in their own language."
        if origin else
        f"Search in English across major music press (Pitchfork, Rolling Stone, NME, Genius, AllMusic). "
        f"If the artist is non-English-speaking, also search in their native language."
    )

    prompt = f"""Research the song "{song_title}" by {primary} for a music podcast.

{native_lang_instruction}

Find and summarize concisely (2-3 paragraphs max):
- What the song is about: lyrics, themes, meaning — including cultural or linguistic nuance for non-English artists
- Has the artist discussed this song in interviews? Any quotes about writing or recording it? (translate if needed)
- Recording context: which album, when, any notable production details or collaborators
- Critical reception or cultural significance, including in the artist's home country

Focus on specific facts and quotes. Keep it short if there isn't much to find."""

    result: dict = {
        "track_id": spotify_track_id,
        "artist": artist_name,
        "title": song_title,
        "research": "",
    }
    try:
        text = _grounded_query(prompt)
        if text:
            result["research"] = text
    except Exception as exc:
        logger.warning("Grounded research failed for song %s - %s: %s", song_title, primary, exc)

    # Only cache successful results — empty/failed results should be retried next run
    if result["research"]:
        cache.put(_CACHE_PREFIX, key, result)
    return result


def _research_album(artist_name: str, album_title: str) -> dict:
    """Research an album using Gemini + Google Search. Keyed by artist + album title."""
    key = cache.cache_key("album_v1", artist_name, album_title)
    cached = cache.get(_CACHE_PREFIX, key)
    if cached is not None:
        return cached

    primary = artist_name.split(",")[0].strip()
    origin = _get_artist_origin(primary)
    native_lang_instruction = (
        f"The artist's origin: {origin}\n"
        f"Issue searches in BOTH English AND the artist's native language — you need both.\n"
        f"English searches: look in Pitchfork, Rolling Stone, NME, AllMusic, The Guardian, and international press "
        f"for reviews and artist interviews about this album.\n"
        f"Native-language searches: search the album title in the native language, check local music press "
        f"and any in-depth interviews about the making of this record."
        if origin else
        f"Search in English across major music press (Pitchfork, Rolling Stone, NME, AllMusic). "
        f"If the artist is non-English-speaking, also search in their native language."
    )

    prompt = f"""Research the album "{album_title}" by {primary} for a music podcast.

{native_lang_instruction}

Find and summarize concisely (2-4 paragraphs):
- The album's themes, concept, or artistic vision — what was the artist trying to express?
- Recording and production: where/when was it made, who produced it, any notable studio stories or techniques
- Has the artist discussed this album in interviews? Key quotes about its creation or meaning (translate if needed)
- Critical reception: how was it reviewed on release? Any awards, legacy, or cultural impact?
- Context in the artist's career: how does it fit relative to their other work?

Focus on specific facts, quotes, and stories. Keep it short if there isn't much to find."""

    result: dict = {"artist": artist_name, "album": album_title, "research": ""}
    try:
        text = _grounded_query(prompt, max_output_tokens=2000)
        if text:
            result["research"] = text
    except Exception as exc:
        logger.warning("Grounded research failed for album %s - %s: %s", album_title, primary, exc)

    if result["research"]:
        cache.put(_CACHE_PREFIX, key, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Parallel enrichment
# ─────────────────────────────────────────────────────────────────────────────

def enrich_with_research(
    context: dict[str, Any],
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Research all artists and songs via Gemini + Google Search grounding.

    Populates:
      - context["artist_profiles"][name]["research"]
      - track["song_research"] for each track

    Results are disk-cached so subsequent runs skip network calls.
    """
    artist_profiles: dict[str, dict] = context.get("artist_profiles", {})
    tracks: list[dict] = context.get("tracks", [])

    # Deduplicate albums: keyed by (primary_artist, album_title)
    seen_albums: set[tuple[str, str]] = set()
    album_tasks: list[tuple[str, str]] = []
    for track in tracks:
        primary = track["artist"].split(",")[0].strip()
        album = track.get("album", "").strip()
        if album and (primary, album) not in seen_albums:
            seen_albums.add((primary, album))
            album_tasks.append((primary, album))

    tasks: list[tuple[str, tuple]] = []
    for artist_name in artist_profiles:
        tasks.append(("artist", (artist_name,)))
    for track in tracks:
        track_id = track["uri"].split(":")[-1]
        tasks.append(("song", (track["artist"], track["name"], track_id)))
    for primary, album in album_tasks:
        tasks.append(("album", (primary, album)))

    def _cache_key_for(kind: str, args: tuple) -> str:
        if kind == "artist":
            return cache.cache_key("artist_v3", args[0])
        elif kind == "song":
            return cache.cache_key("song_v3", args[2])
        else:
            return cache.cache_key("album_v1", args[0], args[1])

    cached_count = sum(
        1 for kind, args in tasks
        if cache.get(_CACHE_PREFIX, _cache_key_for(kind, args)) is not None
    )
    total = len(tasks)
    uncached = total - cached_count

    msg = f"Researching: {total} items ({cached_count} cached, {uncached} to fetch)..."
    logger.info("[Research] %s", msg)
    if progress_cb:
        progress_cb(msg)

    results_by_key: dict[tuple, dict] = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {}
        for task_type, args in tasks:
            if task_type == "artist":
                fut = pool.submit(_research_artist, *args)
            elif task_type == "song":
                fut = pool.submit(_research_song, *args)
            else:
                fut = pool.submit(_research_album, *args)
            futures[fut] = (task_type, args)

        for fut in as_completed(futures):
            task_type, args = futures[fut]
            completed += 1
            if task_type == "artist":
                label = f"artist: {args[0]}"
            elif task_type == "album":
                label = f"album: {args[1]} — {args[0]}"
            else:
                label = f"song: {args[1]} — {args[0].split(',')[0].strip()}"
            try:
                result = fut.result()
                results_by_key[(task_type, args)] = result
                found = bool(result.get("research"))
                status = "+" if found else "-"
                msg = f"Research {completed}/{total}: {status} {label}"
            except Exception as exc:
                logger.warning("Research task failed %s %s: %s", task_type, args, exc)
                msg = f"Research {completed}/{total}: failed — {label}"

            logger.info("[Research] %s", msg)
            if progress_cb:
                progress_cb(msg)

    # Write results back into context
    for artist_name, profile in artist_profiles.items():
        res = results_by_key.get(("artist", (artist_name,)), {})
        profile["research"] = res.get("research", "")
        profile["origin"] = res.get("origin", "")

    # Build album research lookup: (primary_artist, album) → research text
    album_research: dict[tuple[str, str], str] = {}
    for primary, album in album_tasks:
        res = results_by_key.get(("album", (primary, album)), {})
        album_research[(primary, album)] = res.get("research", "")

    for track in tracks:
        track_id = track["uri"].split(":")[-1]
        res = results_by_key.get(("song", (track["artist"], track["name"], track_id)), {})
        track["song_research"] = res.get("research", "")
        primary = track["artist"].split(",")[0].strip()
        track["album_research"] = album_research.get((primary, track.get("album", "")), "")

    hits = sum(1 for p in artist_profiles.values() if p.get("research"))
    song_hits = sum(1 for t in tracks if t.get("song_research"))
    album_hits = sum(1 for v in album_research.values() if v)
    summary = (
        f"Research complete: {hits}/{len(artist_profiles)} artists, "
        f"{album_hits}/{len(album_tasks)} albums, {song_hits}/{len(tracks)} songs"
    )
    logger.info("[Research] %s", summary)
    if progress_cb:
        progress_cb(summary)

    return context
