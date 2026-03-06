"""Gemini script generation — Google AI Studio (API key) or Vertex AI (ADC)."""

from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types

from spoticast import cache
from spoticast.config import settings

_CACHE_PREFIX = "gemini"

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is not None:
        return _client
    if settings.gemini_api_key:
        _client = genai.Client(api_key=settings.gemini_api_key)
    else:
        import google.auth
        project = settings.google_cloud_project
        if not project:
            _, project = google.auth.default()
        _client = genai.Client(
            vertexai=True,
            project=project,
            location=settings.google_cloud_location,
        )
    return _client


_SYSTEM_PROMPT = """You are a brilliant, deeply knowledgeable music podcast duo:

HOST_A — "Sam": The intellectual lead. Analytically sharp, culturally wide-ranging, and genuinely authoritative. Drives the conversation — sets up topics, lands the key insight, frames why something matters. Draws on music history, criticism, and cultural context. Comfortable with complexity and not afraid to be definitive.

HOST_B — "Alex": Warm, enthusiastic, loves the feel and texture of music — production details, what a song does to you, the weird human stories behind recordings. Reacts genuinely to what Sam says, asks the questions a curious listener would ask, and occasionally surprises with an obscure detail he's fixated on.

Together you create rich, intelligent, genuinely engaging DJ commentary for a Spotify playlist — the kind you'd hear on a late-night public radio show hosted by people who really know and love music.

CORE PRINCIPLES:
- Write for spoken word: natural contractions, conversational rhythm, occasional wit.
- Vary sentence length — short punchy lines followed by longer flowing ones.
- Each line should earn its place — every sentence should be either informative, interesting, or emotionally resonant.
- FOCUS ON THE ARTISTS AND THEIR STORIES. This is a music knowledge show, not a listener profile show.

CONVERSATIONAL TEXTURE — this is the most important instruction:
The hosts are in a real conversation. They react, interrupt, disagree, finish each other's thoughts, and build on what was just said. Each line should feel like a response to the previous one, not a prepared monologue taking turns.

TECHNIQUES to use freely:
- Picking up a specific word or idea from the previous line: "Right, and that's exactly what's strange about it —"
- Gentle disagreement or a different angle: "I'd actually push back on that a little. For me it's less about X and more about Y."
- Adding a detail that sharpens what the other said: "And what makes that even more remarkable is—"
- A short interjection before developing the thought: "Wait, actually—" / "That's the thing though—" / "Which is wild because—"
- One host asking the other a genuine question mid-discussion: "Do you think he knew that at the time?"
- Surprise at what the other just said: "I didn't know that." / "That changes how I hear the whole record."
- One host landing the closing thought after the other set it up

SOUND HUMAN — imperfection is realism:
Real people talking about music don't speak in polished paragraphs. Use these sparingly but deliberately:
- Filler and hesitation: "I mean—", "it's like—", "sort of—", "I want to say...", "what's the word—", "kind of, almost—"
- Trailing off and self-correcting: "No, wait—actually that's not quite right.", "Or—well, maybe not, but—"
- Genuine uncertainty about a fact: "I think it was recorded in '94? Maybe '95.", "I could be wrong about this, but—"
- Losing the thread slightly before recovering: "There's something about—I keep coming back to the production on this. The drums specifically."
- One host talking over themselves as they find the thought: "It's—it does this thing where—okay, the bridge."
Use these maybe once or twice per commentary block, not constantly. The imperfection should feel accidental, not performed.

NEVER do this:
- Two consecutive lines where both hosts deliver independent paragraphs of fact with no reference to each other
- Filler affirmations: "Absolutely", "Great point", "Totally", "Yeah, exactly", "That's so true", "100%"
- Polite turn-taking that sounds like two people reading from separate scripts
- Overdoing the imperfection — one stumble per exchange is enough; this isn't a parody

INTRO (~5 minutes, 10-16 exchanges):
- Open with something surprising or evocative — a fact, an unlikely connection, or a sharp observation about the music itself. Never "welcome to our show."
- Discuss what makes the artists in this playlist interesting: their origins, their cultural moment, what they represent
- Draw unexpected connections between artists in the playlist — shared influences, parallel careers, musical debts
- Surface compelling trivia: recording stories, feuds, failures, breakthroughs, strange facts about how these songs were made
- Create genuine anticipation for what's coming
- DO NOT discuss the first track in depth — that gets its own dedicated commentary segment immediately after. You can mention it briefly at most.

PER-TRACK COMMENTARY (5-8 exchanges per track, 60-120 seconds when spoken):
- Open by naturally naming the song and artist the listener just heard — e.g. "That was [Song] by [Artist]" or woven into a reaction: "There's something about the way [Artist] ends [Song] that always gets me." Do not make it a stiff announcement; let it feel like two people coming out of the music.
- Then transition to the NEXT track: name the specific connection (tempo, key, mood, era, artist relationship, thematic link, lyrical echo)
- What is the NEXT song ABOUT: the lyrics, the imagery, the emotional core. Be specific — quote or paraphrase a line if it's illuminating.
- What has the NEXT TRACK'S ARTIST said about this song? Draw on interviews, statements, and known context. Specific anecdotes beat vague praise.
- Production story, recording context, what album it came from and why that album matters
- Surprising trivia: a fact most listeners don't know. Something that changes how you hear the song.
- End with something that creates anticipation for the music itself — let the song start with intention

USING THE DATA PROVIDED:
- Artist bio, tags, similar artists: use as starting points for stories, not as labels to recite
- Research notes (interviews, album context, recording stories): these are your raw material — surface the most compelling detail
- Last.fm wiki blurbs and song research: mine for specific facts, then weave them in naturally
- Audio features: use impressionistically ("there's a stillness to this recording") not numerically
- Listener data (scrobbles, loved tracks, fan era): use sparingly and only when it reveals something interesting about the music itself, not to comment on listening habits

Output ONLY valid JSON — no preamble, no explanation, no markdown."""

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "intro": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "host": {"type": "string", "enum": ["HOST_A", "HOST_B"]},
                    "text": {"type": "string"},
                },
                "required": ["host", "text"],
            },
        },
        "tracks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "track_uri": {"type": "string"},
                    "commentary": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "host": {"type": "string", "enum": ["HOST_A", "HOST_B"]},
                                "text": {"type": "string"},
                            },
                            "required": ["host", "text"],
                        },
                    },
                },
                "required": ["track_uri", "commentary"],
            },
        },
    },
    "required": ["intro", "tracks"],
}


def build_prompt(context: dict[str, Any]) -> str:
    summary    = context["summary"]
    profile    = context["listener_profile"]
    tracks     = context["tracks"]
    artist_profiles: dict[str, dict] = context.get("artist_profiles", {})
    lastfm_user: dict = context.get("lastfm_user", {})

    has_lastfm = bool(lastfm_user)

    # ── Listener overview ────────────────────────────────────────────────────
    listener_lines: list[str] = []
    if has_lastfm:
        listener_lines.append(
            f"Last.fm user '{lastfm_user.get('username')}' — "
            f"{lastfm_user.get('total_scrobbles', '?')} total scrobbles, "
            f"listening since {lastfm_user.get('member_since', '?')}"
        )
    top_artists = (
        profile.get("lastfm_top_artists")
        or list(profile.get("top_artists_all_time", []))
    )
    if top_artists:
        listener_lines.append("Top artists all-time: " + ", ".join(top_artists[:15]))

    if summary.get("lastfm_total_plays"):
        listener_lines.append(
            f"Total plays of playlist tracks on Last.fm: {summary['lastfm_total_plays']}"
        )

    # ── Artist profiles ──────────────────────────────────────────────────────
    artist_section_lines: list[str] = []
    for artist_name, ap in artist_profiles.items():
        lines = [f"\n{artist_name}:"]
        if ap.get("origin"):
            lines.append(f"  Origin: {ap['origin']}")
        if ap.get("fan_era"):
            lines.append(f"  Listener relationship: {ap['fan_era']}")
        if ap.get("tags"):
            lines.append(f"  Tags/genre: {', '.join(ap['tags'][:5])}")
        if ap.get("similar_you_love"):
            lines.append(f"  Similar artists you also love: {', '.join(ap['similar_you_love'][:4])}")
        if ap.get("bio_summary"):
            lines.append(f"  Bio: {ap['bio_summary']}")
        if ap.get("research"):
            lines.append(f"  Research: {ap['research']}")
        artist_section_lines.extend(lines)

    # ── Track list ───────────────────────────────────────────────────────────
    def _track_block(i: int, t: dict) -> str:
        feat = t.get("features") or {}
        album_str = f", album: {t['album']}" if t.get("album") else ""
        lines = [
            f"\n{i+1}. \"{t['name']}\" — {t['artist']} ({t['year']}{album_str}, {t['duration_s']}s)"
        ]

        # Audio character (impressionistic, not numeric dump)
        if feat:
            energy_word = (
                "high energy" if feat.get("energy", 0) > 0.7
                else "mid-energy" if feat.get("energy", 0) > 0.4
                else "quiet/delicate"
            )
            mood_word = (
                "bright/uplifting" if feat.get("valence", 0) > 0.65
                else "bittersweet" if feat.get("valence", 0) > 0.35
                else "dark/melancholic"
            )
            lines.append(f"  Sound: {energy_word}, {mood_word}, {feat.get('tempo', '?')} BPM, {feat.get('key', '?')} {feat.get('mode', '')}")

        # Listener signals
        signals: list[str] = []
        if t.get("lastfm_loved"):
            signals.append("LOVED by listener on Last.fm")
        elif t.get("is_personal_favorite"):
            signals.append("in Spotify top tracks")
        if t.get("lastfm_play_count", 0) > 0:
            signals.append(f"{t['lastfm_play_count']} Last.fm scrobbles")
        if t.get("artist_in_top"):
            signals.append("artist is a top artist")
        if signals:
            lines.append("  Listener: " + " | ".join(signals))

        # Tags
        if t.get("tags"):
            lines.append("  Tags: " + ", ".join(t["tags"][:5]))

        # Last.fm wiki blurb
        if t.get("lastfm_wiki"):
            lines.append(f"  Last.fm: {t['lastfm_wiki']}")

        # Album research (themes, recording context, reception)
        if t.get("album_research"):
            lines.append(f"  Album research: {t['album_research']}")

        # Song research (interviews, recording stories, context)
        if t.get("song_research"):
            lines.append(f"  Song research: {t['song_research']}")

        lines.append(f"  URI: {t['uri']}")
        return "\n".join(lines)

    track_section = "\n".join(_track_block(i, t) for i, t in enumerate(tracks))

    return f"""Generate a rich, detailed podcast commentary script for this Spotify playlist.

═══ LISTENER PROFILE ═══
{chr(10).join(listener_lines)}

═══ PLAYLIST OVERVIEW ═══
{summary['total_tracks']} tracks | Avg energy: {summary['avg_energy']} | Avg valence: {summary['avg_valence']} | Avg tempo: {summary['avg_tempo']} BPM
{summary['personal_favorites_count']} tracks are Spotify top tracks | {summary.get('lastfm_total_plays', 0)} total Last.fm plays across playlist

═══ ARTIST PROFILES ═══
(Use these for trivia, connections, fan relationship context)
{''.join(artist_section_lines)}

═══ PLAYLIST TRACKS (in order) ═══
(Each track's commentary opens by naming the song/artist just heard, then pivots to the next track: what it's about, artist context, production story, surprising trivia)
{track_section}

Generate the complete podcast script. Make the per-track commentary genuinely rich — 5-8 exchanges, 60-120 seconds when spoken. Use the research data above as concrete talking points, not just background.
"""


def generate_episode_name(context: dict[str, Any]) -> str:
    """
    Generate a short, evocative podcast episode title from the playlist context.

    Uses the research model (Flash) for speed — this is a lightweight creative task.
    Returns a 2-5 word title, e.g. "Berlin Nights", "The Britpop Years".
    """
    track_lines = [
        f"{t['name']} — {t['artist']}"
        for t in context.get("tracks", [])[:15]
    ]
    playlist_name = context.get("playlist_name", "")

    prompt = (
        f"Generate a short podcast episode title for a music commentary show.\n"
        f"Playlist: {playlist_name}\n"
        f"Tracks include:\n" + "\n".join(f"  - {l}" for l in track_lines) + "\n\n"
        "Requirements:\n"
        "- 2–5 words maximum\n"
        "- Evocative and specific — capture the mood, era, geography, or emotional thread\n"
        "- Like a real episode title: 'Berlin Nights', 'Melancholy at Midnight', 'The Britpop Years'\n"
        "- No quotes, no punctuation except hyphens\n"
        "Output ONLY the title, nothing else."
    )

    response = _get_client().models.generate_content(
        model=settings.gemini_research_model,
        contents=prompt,
    )
    return response.text.strip().strip("\"'")


def generate_script(context: dict[str, Any]) -> dict:
    """Call Gemini and return the parsed script JSON."""
    # Cache key: sorted track URIs + Last.fm username (identifies playlist + listener)
    track_uris = sorted(t["uri"] for t in context["tracks"])
    username = context.get("lastfm_user", {}).get("username", "")
    script_key = cache.cache_key("script_v1", username, *track_uris)
    cached = cache.get(_CACHE_PREFIX, script_key)
    if cached is not None:
        return cached

    response = _get_client().models.generate_content(
        model=settings.gemini_model,
        contents=build_prompt(context),
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
            max_output_tokens=32000,
            temperature=1.1,
        ),
    )

    result = json.loads(response.text)
    cache.put(_CACHE_PREFIX, script_key, result)
    return result
