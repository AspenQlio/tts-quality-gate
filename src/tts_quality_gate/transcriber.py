from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Transcriber(Protocol):
    """Protocol for transcribing generated audio back to text."""

    def transcribe(self, audio_path: str) -> str:
        """Returns the transcribed text from the audio file."""
        ...


class MockTranscriber:
    """A determinist transcriber for testing without heavy ML models."""

    def __init__(self, forced_responses: dict[str, str] | None = None) -> None:
        self.forced_responses = forced_responses or {}

    def transcribe(self, audio_path: str) -> str:
        return self.forced_responses.get(audio_path, "mocked transcription")


def build_transcriber(backend: str) -> Transcriber:
    if backend == "mock":
        return MockTranscriber()
    raise ValueError(f"Unknown transcription backend: {backend}")
