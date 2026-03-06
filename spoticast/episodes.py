"""Episode persistence — each generated spoticast is stored as a JSON manifest."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

_EPISODES_DIR = Path("generated") / "episodes"


def _ensure_dir() -> Path:
    _EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    return _EPISODES_DIR


def episode_audio_dir(episode_id: str) -> str:
    """Return the relative path (from generated/) for an episode's audio files."""
    return f"episodes/{episode_id}"


def save_episode(
    episode_id: str,
    name: str,
    playlist_uri: str,
    playlist_name: str,
    track_count: int,
    queue: list[dict],
) -> None:
    _ensure_dir()
    meta = {
        "id": episode_id,
        "name": name,
        "playlist_uri": playlist_uri,
        "playlist_name": playlist_name,
        "track_count": track_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "queue": queue,
    }
    path = _EPISODES_DIR / episode_id / "episode.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))


def list_episodes() -> list[dict]:
    """Return episode summaries (no queue), newest first."""
    if not _EPISODES_DIR.exists():
        return []
    episodes = []
    for ep_dir in _EPISODES_DIR.iterdir():
        meta_path = ep_dir / "episode.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        episodes.append({
            "id": meta["id"],
            "name": meta["name"],
            "playlist_uri": meta.get("playlist_uri", ""),
            "playlist_name": meta.get("playlist_name", ""),
            "track_count": meta.get("track_count", 0),
            "created_at": meta["created_at"],
        })
    episodes.sort(key=lambda e: e["created_at"], reverse=True)
    return episodes


def get_episode(episode_id: str) -> dict | None:
    path = _EPISODES_DIR / episode_id / "episode.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
