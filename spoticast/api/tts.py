"""Gemini TTS — multi-speaker dialogue synthesis via Vertex AI (ADC auth)."""

from __future__ import annotations

from google import genai
from google.genai import types

from spoticast.config import settings

_client: genai.Client | None = None

# Speaker name and voice for each host role.
# Charon (Informative) suits the analytical HOST_A; Aoede (Breezy) fits the intuitive HOST_B.
_VOICE_MAP = {
    "HOST_A": ("Alex", "Charon"),
    "HOST_B": ("Sam", "Aoede"),
}


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if settings.gemini_api_key:
            _client = genai.Client(api_key=settings.gemini_api_key)
        else:
            import google.auth
            project = settings.google_cloud_project
            if not project:
                _, project = google.auth.default()
            # TTS models require a regional endpoint, not "global"
            _client = genai.Client(vertexai=True, project=project, location="us-central1")
    return _client


def synthesize_dialogue(lines: list[dict]) -> bytes:
    """
    Synthesize a full multi-speaker dialogue in one API call.

    Returns raw PCM bytes: 16-bit signed, 24kHz, mono.
    The model handles natural pacing and speaker transitions.
    """
    client = _get_client()

    # Format the dialogue so the model knows which speaker says each line.
    # Speaker names must match the keys in speaker_voice_configs below.
    dialogue_parts = []
    for line in lines:
        name, _ = _VOICE_MAP.get(line["host"], _VOICE_MAP["HOST_A"])
        dialogue_parts.append(f"{name}: {line['text']}")

    # Collect the unique speaker names present in this dialogue block
    speakers_present = sorted({_VOICE_MAP[l["host"]][0] for l in lines if l["host"] in _VOICE_MAP})
    speaker_list = " and ".join(speakers_present)
    prompt = f"TTS the following conversation between {speaker_list}:\n" + "\n".join(dialogue_parts)

    speaker_voice_configs = [
        types.SpeakerVoiceConfig(
            speaker=name,
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            ),
        )
        for _host, (name, voice) in _VOICE_MAP.items()
    ]

    response = client.models.generate_content(
        model="gemini-2.5-flash-tts",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=speaker_voice_configs,
                ),
            ),
        ),
    )

    return response.candidates[0].content.parts[0].inline_data.data
