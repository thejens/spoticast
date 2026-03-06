"""
Last.fm enrichment — deep listener context for AI commentary.

Fetches:
  - User profile (scrobble count, member since)
  - Multi-period top artist rankings → fan-era classification per artist
  - Per-playlist-artist: bio, tags, global stats, similar artists × user affinity
  - Per-playlist-track: user play count, loved status, tags, wiki blurb
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pylast
from dataclasses import asdict

from spoticast import cache
from spoticast.config import settings

_CACHE_PREFIX = "lastfm"

logger = logging.getLogger(__name__)

# In-memory caches — survive across regenerations within the same process
_artist_cache: dict[str, ArtistProfile] = {}   # normalized name → profile
_track_enrich_cache: dict[str, dict] = {}       # "artist|title" → enrichment dict
_rankings_cache: dict[str, ArtistRankings] = {} # username → rankings
_loved_cache: dict[str, set[str]] = {}          # username → loved track keys
_top_plays_cache: dict[str, dict[str, int]] = {} # username → track key → play count

# Persists the username across restarts without touching .env
_USER_FILE = Path(".lastfm_user")

# Runtime-override — set via connect() and survives the process lifetime
_runtime_username: str | None = None


def get_username() -> str | None:
    """Return the effective Last.fm username: runtime > file > env."""
    if _runtime_username:
        return _runtime_username
    if _USER_FILE.exists():
        u = _USER_FILE.read_text().strip()
        if u:
            return u
    return settings.lastfm_username or None


def is_configured() -> bool:
    return bool(settings.lastfm_api_key and settings.lastfm_api_secret and get_username())


def connect(username: str) -> dict:
    """
    Validate a Last.fm username and persist it.
    Returns {"username": str, "scrobbles": int, "registered": str} on success.
    Raises ValueError on unknown username.
    """
    global _runtime_username
    network = _get_network()
    user = network.get_user(username)
    try:
        scrobbles = int(user.get_playcount())
        registered = user.get_registered()
        import datetime
        if isinstance(registered, (int, float)):
            registered_str = str(datetime.datetime.fromtimestamp(registered).year)
        else:
            registered_str = str(registered)[:4]
    except pylast.WSError as exc:
        raise ValueError(f"Last.fm user '{username}' not found") from exc

    _runtime_username = username
    _USER_FILE.write_text(username)

    return {"username": username, "scrobbles": scrobbles, "registered": registered_str}


def disconnect() -> None:
    global _runtime_username
    _runtime_username = None
    if _USER_FILE.exists():
        _USER_FILE.unlink()
    _rankings_cache.clear()
    _loved_cache.clear()
    _top_plays_cache.clear()


def get_status() -> dict:
    """
    Returns Last.fm status for the frontend.

    `available` — API keys are present in config (widget should be shown).
    `connected` — a username is also set and validated.
    """
    has_keys = bool(settings.lastfm_api_key and settings.lastfm_api_secret)
    if not has_keys:
        return {"available": False, "connected": False, "username": None}

    username = get_username()
    if not username:
        return {"available": True, "connected": False, "username": None}

    try:
        network = _get_network()
        user = network.get_user(username)
        scrobbles = int(user.get_playcount())
        return {"available": True, "connected": True, "username": username, "scrobbles": scrobbles}
    except Exception:
        return {"available": True, "connected": True, "username": username, "scrobbles": None}


def _get_network() -> pylast.LastFMNetwork:
    return pylast.LastFMNetwork(
        api_key=settings.lastfm_api_key,
        api_secret=settings.lastfm_api_secret,
    )


def _normalize(s: str) -> str:
    return s.strip().lower()


def _track_key(artist: str, title: str) -> str:
    return f"{_normalize(artist.split(',')[0])}|{_normalize(title)}"


def _strip_html(text: str) -> str:
    import re
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Truncate at "Read more on" links that Last.fm appends
    for marker in ["Read more on Last.fm", "User-contributed text"]:
        if marker in text:
            text = text[:text.index(marker)].strip()
    return text


def _first_sentences(text: str, n: int = 2) -> str:
    """Return the first n sentences of text."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:n])


# ─────────────────────────────────────────────────────────────────────────────
# Fan-era classification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArtistRankings:
    """Artist name → play count across time periods."""
    overall:  dict[str, int] = field(default_factory=dict)
    year:     dict[str, int] = field(default_factory=dict)
    quarter:  dict[str, int] = field(default_factory=dict)
    month:    dict[str, int] = field(default_factory=dict)
    # overall artist name → rank position (1-indexed)
    overall_rank: dict[str, int] = field(default_factory=dict)


def _fetch_period_rankings(user: pylast.User) -> ArtistRankings:
    """Fetch top artists across all time periods in one pass."""
    rankings = ArtistRankings()
    # overall
    try:
        items = user.get_top_artists(period="overall", limit=500)
        for rank, item in enumerate(items, 1):
            name = _normalize(item.item.name)
            rankings.overall[name] = int(item.weight)
            rankings.overall_rank[name] = rank
    except Exception as exc:
        logger.warning("Could not fetch overall top artists: %s", exc)

    for period_key, attr, limit in [
        ("12month", "year",    200),
        ("3month",  "quarter", 100),
        ("1month",  "month",    50),
    ]:
        try:
            items = user.get_top_artists(period=period_key, limit=limit)
            target = getattr(rankings, attr)
            for item in items:
                target[_normalize(item.item.name)] = int(item.weight)
        except Exception as exc:
            logger.warning("Could not fetch %s top artists: %s", period_key, exc)

    return rankings


def _classify_fan_era(artist_name: str, r: ArtistRankings) -> str:
    name = _normalize(artist_name)
    in_overall  = name in r.overall
    in_year     = name in r.year
    in_quarter  = name in r.quarter
    in_month    = name in r.month
    rank        = r.overall_rank.get(name)
    plays       = r.overall.get(name, 0)

    if in_month and in_overall:
        if rank and rank <= 30:
            return f"longtime deep fan (#{rank} all-time, {plays} plays) — currently active"
        return f"rediscovered recently ({plays} all-time plays)"
    if in_month and not in_year:
        return "recent discovery (new in the last month)"
    if in_quarter and not in_year:
        return "current phase (past 3 months)"
    if in_year and in_overall:
        if rank and rank <= 50:
            return f"consistent longtime fan (#{rank} all-time, {plays} plays)"
        return f"regular listener ({plays} all-time plays)"
    if in_overall and not in_year:
        if plays and plays >= 200:
            return f"past obsession ({plays} plays, less active now)"
        return "past listen, rarely plays now"
    if in_overall:
        return f"occasional listener ({plays} plays)"
    return "not a regular listen"


# ─────────────────────────────────────────────────────────────────────────────
# Artist deep-fetch
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArtistProfile:
    name: str
    fan_era: str
    user_play_count: int
    global_listeners: int
    tags: list[str]
    bio_summary: str          # Last.fm bio, 2–3 sentences
    similar: list[str]        # similar artist names
    similar_you_love: list[str]  # similar artists the user also listens to (with play counts)


def _fetch_artist_profile(
    network: pylast.LastFMNetwork,
    artist_name: str,
    rankings: ArtistRankings,
) -> ArtistProfile:
    name_lower = _normalize(artist_name)
    if name_lower in _artist_cache:
        cached = _artist_cache[name_lower]
        # Re-classify fan era with current rankings (cheap)
        cached.fan_era = _classify_fan_era(artist_name, rankings)
        return cached

    # Try disk cache before hitting the API
    disk_key = cache.cache_key("artist", artist_name)
    disk_data = cache.get(_CACHE_PREFIX, disk_key)
    if disk_data is not None:
        profile = ArtistProfile(**disk_data)
        profile.fan_era = _classify_fan_era(artist_name, rankings)
        _artist_cache[name_lower] = profile
        return profile

    user_play_count = rankings.overall.get(name_lower, 0)
    fan_era = _classify_fan_era(artist_name, rankings)

    artist = network.get_artist(artist_name)

    tags: list[str] = []
    try:
        raw_tags = artist.get_top_tags(limit=8)
        tags = [t.item.name for t in raw_tags if int(t.weight) >= 20][:6]
    except Exception:
        pass

    bio_summary = ""
    try:
        raw_bio = artist.get_bio_summary()
        if raw_bio:
            bio_summary = _first_sentences(_strip_html(raw_bio), 3)
    except Exception:
        pass

    global_listeners = 0
    try:
        global_listeners = artist.get_listener_count()
    except Exception:
        pass

    similar_names: list[str] = []
    similar_you_love: list[str] = []
    try:
        similar_items = artist.get_similar(limit=25)
        for s in similar_items:
            sname = s.item.name
            similar_names.append(sname)
            sname_lower = _normalize(sname)
            if sname_lower in rankings.overall:
                plays = rankings.overall[sname_lower]
                similar_you_love.append(f"{sname} ({plays} plays)")
    except Exception:
        pass

    profile = ArtistProfile(
        name=artist_name,
        fan_era=fan_era,
        user_play_count=user_play_count,
        global_listeners=global_listeners,
        tags=tags,
        bio_summary=bio_summary,
        similar=similar_names[:10],
        similar_you_love=similar_you_love[:6],
    )
    cache.put(_CACHE_PREFIX, disk_key, asdict(profile))
    _artist_cache[name_lower] = profile
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Track enrichment
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_tracks(
    network: pylast.LastFMNetwork,
    tracks: list[dict],
    top_track_plays: dict[str, int],
    loved_keys: set[str],
) -> None:
    """Add lastfm_play_count, lastfm_loved, tags, wiki_blurb to each track dict in-place."""
    for track in tracks:
        key = _track_key(track["artist"], track["name"])
        track["lastfm_play_count"] = top_track_plays.get(key, 0)
        track["lastfm_loved"] = key in loved_keys

        # Check in-memory then disk cache for expensive per-track API calls (tags, wiki)
        if key in _track_enrich_cache:
            cached = _track_enrich_cache[key]
            track["tags"] = cached["tags"]
            track["lastfm_wiki"] = cached["lastfm_wiki"]
            continue

        disk_key = cache.cache_key("track_enrich", key)
        disk_data = cache.get(_CACHE_PREFIX, disk_key)
        if disk_data is not None:
            track["tags"] = disk_data["tags"]
            track["lastfm_wiki"] = disk_data["lastfm_wiki"]
            _track_enrich_cache[key] = disk_data
            continue

        tags: list[str] = []
        wiki_blurb = ""
        try:
            lfm_track = network.get_track(
                track["artist"].split(",")[0].strip(), track["name"]
            )
            raw_tags = lfm_track.get_top_tags(limit=6)
            tags = [t.item.name for t in raw_tags if int(t.weight) >= 15][:5]
            try:
                raw_wiki = lfm_track.get_wiki_summary()
                if raw_wiki:
                    wiki_blurb = _first_sentences(_strip_html(raw_wiki), 2)
            except Exception:
                pass
        except Exception:
            pass

        enriched = {"tags": tags, "lastfm_wiki": wiki_blurb}
        track["tags"] = tags
        track["lastfm_wiki"] = wiki_blurb
        _track_enrich_cache[key] = enriched
        cache.put(_CACHE_PREFIX, cache.cache_key("track_enrich", key), enriched)


# ─────────────────────────────────────────────────────────────────────────────
# User profile
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    username: str
    total_scrobbles: int
    member_since: str   # year string


def _fetch_user_profile(user: pylast.User) -> UserProfile:
    username: str = get_username() or ""
    total_scrobbles = 0
    member_since = "unknown"
    try:
        info = user.get_registered()  # returns datetime or timestamp
        if info:
            import datetime
            if isinstance(info, (int, float)):
                member_since = str(datetime.datetime.fromtimestamp(info).year)
            else:
                member_since = str(info)[:4]
    except Exception:
        pass
    try:
        total_scrobbles = int(user.get_playcount())
    except Exception:
        pass
    return UserProfile(username=username, total_scrobbles=total_scrobbles, member_since=member_since)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def enrich_context(context: dict[str, Any], progress_cb=None) -> dict[str, Any]:
    """
    Enrich the playlist context with Last.fm data in-place and return it.

    Adds:
      - context["lastfm_user"]: user profile dict
      - context["artist_profiles"]: per-artist dict with fan era, bio, similar artists
      - Per-track: lastfm_play_count, lastfm_loved, tags, lastfm_wiki
      - context["summary"]["lastfm_total_plays"]

    progress_cb: optional callable(message: str) for streaming progress updates.
    """
    def _progress(msg: str):
        logger.info("[Last.fm] %s", msg)
        if progress_cb:
            progress_cb(msg)

    network = _get_network()
    username = get_username() or ""
    user = network.get_user(username)

    # ── User profile ─────────────────────────────────────────────────────────
    _progress("Fetching user profile...")
    profile = _fetch_user_profile(user)
    context["lastfm_user"] = {
        "username": profile.username,
        "total_scrobbles": profile.total_scrobbles,
        "member_since": profile.member_since,
    }
    _progress(f"User: {profile.username} — {profile.total_scrobbles:,} scrobbles since {profile.member_since}")

    # ── Multi-period rankings (4 API calls, cached per user) ────────────────
    _rankings_disk_key = cache.cache_key("rankings", username)
    if username in _rankings_cache:
        _progress("Using cached artist rankings...")
        rankings = _rankings_cache[username]
    elif (disk_data := cache.get(_CACHE_PREFIX, _rankings_disk_key)) is not None:
        _progress("Using disk-cached artist rankings...")
        rankings = ArtistRankings(**disk_data)
        _rankings_cache[username] = rankings
    else:
        _progress("Fetching top artists (overall, 12m, 3m, 1m)...")
        rankings = _fetch_period_rankings(user)
        cache.put(_CACHE_PREFIX, _rankings_disk_key, asdict(rankings))
        _rankings_cache[username] = rankings
        _progress(f"Loaded {len(rankings.overall)} artists in your all-time history")

    # ── Loved tracks (cached per user) ────────────────────────────────────────
    _loved_disk_key = cache.cache_key("loved", username)
    if username in _loved_cache:
        _progress("Using cached loved tracks...")
        loved_keys = _loved_cache[username]
    elif (disk_data := cache.get(_CACHE_PREFIX, _loved_disk_key)) is not None:
        _progress("Using disk-cached loved tracks...")
        loved_keys = set(disk_data)
        _loved_cache[username] = loved_keys
    else:
        _progress("Fetching loved tracks...")
        loved_keys: set[str] = set()
        try:
            for item in user.get_loved_tracks(limit=500):
                loved_keys.add(_track_key(item.track.artist.name, item.track.title))
        except Exception as exc:
            logger.warning("Could not fetch loved tracks: %s", exc)
        cache.put(_CACHE_PREFIX, _loved_disk_key, list(loved_keys))
        _loved_cache[username] = loved_keys
        _progress(f"Found {len(loved_keys)} loved tracks")

    # ── Top track play counts (cached per user) ──────────────────────────────
    _top_plays_disk_key = cache.cache_key("top_plays", username)
    if username in _top_plays_cache:
        _progress("Using cached top track play counts...")
        top_track_plays = _top_plays_cache[username]
    elif (disk_data := cache.get(_CACHE_PREFIX, _top_plays_disk_key)) is not None:
        _progress("Using disk-cached top track play counts...")
        top_track_plays = disk_data
        _top_plays_cache[username] = top_track_plays
    else:
        _progress("Fetching top track play counts (up to 1000)...")
        top_track_plays: dict[str, int] = {}
        try:
            for item in user.get_top_tracks(limit=1000, period="overall"):
                key = _track_key(item.item.artist.name, item.item.title)
                top_track_plays[key] = int(item.weight)
        except Exception as exc:
            logger.warning("Could not fetch top tracks: %s", exc)
        cache.put(_CACHE_PREFIX, _top_plays_disk_key, top_track_plays)
        _top_plays_cache[username] = top_track_plays
        _progress(f"Loaded play counts for {len(top_track_plays)} tracks")

    # ── Per-track enrichment ─────────────────────────────────────────────────
    _progress(f"Enriching {len(context['tracks'])} playlist tracks...")
    _enrich_tracks(network, context["tracks"], top_track_plays, loved_keys)

    # ── Per-artist profiles (deduplicated) ───────────────────────────────────
    unique_artists: dict[str, str] = {}   # normalized → original name
    for track in context["tracks"]:
        primary = track["artist"].split(",")[0].strip()
        unique_artists[_normalize(primary)] = primary

    total_artists = len(unique_artists)
    _progress(f"Fetching profiles for {total_artists} artists...")

    artist_profiles: dict[str, dict] = {}
    for i, artist_name in enumerate(unique_artists.values(), 1):
        _progress(f"Building artist profile: {artist_name} ({i}/{total_artists})...")
        try:
            ap = _fetch_artist_profile(network, artist_name, rankings)
            artist_profiles[artist_name] = {
                "fan_era": ap.fan_era,
                "user_play_count": ap.user_play_count,
                "global_listeners": ap.global_listeners,
                "tags": ap.tags,
                "bio_summary": ap.bio_summary,
                "similar_you_love": ap.similar_you_love,
            }
        except Exception as exc:
            logger.warning("Could not fetch artist profile for %s: %s", artist_name, exc)

    context["artist_profiles"] = artist_profiles

    # ── Summary stats ─────────────────────────────────────────────────────────
    total_plays = sum(t.get("lastfm_play_count", 0) for t in context["tracks"])
    context["summary"]["lastfm_total_plays"] = total_plays

    # Scrobble-weighted top artists for the listener profile
    context["listener_profile"]["lastfm_top_artists"] = [
        name for name, _ in sorted(
            rankings.overall.items(), key=lambda x: -x[1]
        )[:20]
    ]

    _progress(f"Last.fm enrichment complete — {total_plays} total plays across playlist tracks")
    return context
