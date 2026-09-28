# TTS Quality Gate

An automated quality gate for Text-to-Speech (TTS) models. It prevents degraded models from reaching production after fine-tuning.

This project implements a continuous integration harness for audio machine learning. It transcribes generated audio using an external transcriber (like Whisper) and calculates the Word Error Rate (WER) against the reference text. A pipeline fails if the aggregate WER exceeds a defined threshold.

## Architecture

The harness evaluates a dataset of test cases in JSONL format. 

1. **Generation:** The fine-tuned TTS model generates an audio file for each reference text.
2. **Transcription:** A transcriber converts the generated audio back into text.
3. **Evaluation:** The runner normalizes both texts (lowercase, no punctuation) and calculates the WER.
4. **Decision:** The pipeline aggregates the WER and applies a strict threshold. It passes only if every case stays under the limit.

## Usage

Define an evaluation dataset (`dataset.jsonl`):

```json
{"id": "test-01", "reference_text": "The server responded with an error.", "audio_path": "outputs/test-01.wav"}
{"id": "test-02", "reference_text": "Restart the database cluster.", "audio_path": "outputs/test-02.wav"}
```

Run the quality gate:

```bash
tts-quality-gate --dataset dataset.jsonl --threshold 0.05 --backend mock --json report.json
```

## Development

```bash
uv sync --all-extras --dev
pytest
ruff check .
```

The test suite covers exact matches, punctuation removal, capitalization invariance, and pipeline threshold enforcement.

## License

MIT
