from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    spotify_client_id: str
    spotify_client_secret: str | None = None

    # Last.fm — enriches listening context with play counts, tags, and loved tracks.
    lastfm_api_key: str | None = None
    lastfm_api_secret: str | None = None
    lastfm_username: str | None = None

    # Gemini — use either an AI Studio API key OR Google Cloud ADC (not both).
    # If GEMINI_API_KEY is set, Vertex AI / GCP credentials are not needed.
    gemini_api_key: str | None = None

    # GCP project for Vertex AI (only used when GEMINI_API_KEY is not set).
    # If unset, resolved automatically from Application Default Credentials.
    google_cloud_project: str | None = None
    google_cloud_location: str = "us-central1"
    gemini_model: str = "gemini-2.5-pro"
    gemini_research_model: str = "gemini-2.5-flash-lite"

    max_tracks: int = 30
    port: int = 8765

    @property
    def redirect_uri(self) -> str:
        return f"http://127.0.0.1:{self.port}/auth/callback"

    @property
    def spotify_scopes(self) -> str:
        return " ".join([
            "user-read-recently-played",
            "user-top-read",
            "playlist-read-private",
            "playlist-read-collaborative",
            "streaming",
            "user-modify-playback-state",
            "user-read-playback-state",
            "user-read-email",
            "user-read-private",
        ])


settings = Settings()
