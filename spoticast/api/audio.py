"""Audio assembly — convert Gemini TTS PCM output to MP3 for each commentary block."""

from __future__ import annotations

from pathlib import Path

from pydub import AudioSegment

# Silence appended after each commentary block, before music resumes
TRAIL_SILENCE_MS = 800

_GENERATED_DIR = Path("generated")


def assemble_commentary(pcm_bytes: bytes, filename: str) -> Path:
    """
    Convert raw PCM dialogue audio to MP3 and write to generated/.

    Input is the raw PCM output from Gemini TTS: 16-bit signed, 24kHz, mono.
    The model already handles natural pacing between speakers, so no inter-speaker
    silence is inserted here — only a short trail before the next music track.
    """
    audio = AudioSegment(data=pcm_bytes, sample_width=2, frame_rate=24000, channels=1)
    audio += AudioSegment.silent(duration=TRAIL_SILENCE_MS)

    output_path = _GENERATED_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(str(output_path), format="mp3")
    return output_path
