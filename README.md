# TTS Quality Gate

> An automated quality gate for Text-to-Speech (TTS) models to prevent degraded audio from reaching production.

TTS Quality Gate implements a continuous integration harness for audio machine learning. It automatically transcribes generated audio using an external transcriber (like Whisper) and calculates the Word Error Rate (WER) against the reference text, failing the pipeline if the error rate exceeds your defined threshold.

## Features

- **Automated Transcription:** Seamlessly converts generated audio back to text for verification.
- **WER Calculation:** Computes the Word Error Rate to ensure high audio fidelity.
- **Strict Validation:** Fails the pipeline automatically if the aggregate WER exceeds the limit.
- **Normalization:** Evaluates text cleanly by handling lowercase conversion and punctuation removal.

## Architecture

The harness evaluates a dataset of test cases in JSONL format through a simple 4-step process:

1. **Generation:** The fine-tuned TTS model generates an audio file for each reference text.
2. **Transcription:** A transcriber converts the generated audio back into text.
3. **Evaluation:** The runner normalizes both texts (lowercase, no punctuation) and calculates the WER.
4. **Decision:** The pipeline aggregates the WER and applies a strict threshold. It passes only if every case stays under the limit.

## Tech Stack

- **Language:** Python
- **Package Management:** uv
- **Testing:** pytest
- **Linting:** ruff

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) installed on your system.

### Installation

Clone the repository and sync the dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/tts-quality-gate.git
cd tts-quality-gate
uv sync --all-extras --dev
```

## Usage

1. Define an evaluation dataset in a `dataset.jsonl` file:

```json
{"id": "test-01", "reference_text": "The server responded with an error.", "audio_path": "outputs/test-01.wav"}
{"id": "test-02", "reference_text": "Restart the database cluster.", "audio_path": "outputs/test-02.wav"}
```

2. Run the quality gate:

```bash
tts-quality-gate --dataset dataset.jsonl --threshold 0.05 --backend mock --json report.json
```

## Development

To run the test suite (which covers exact matches, punctuation removal, capitalization invariance, and pipeline threshold enforcement) and check for linting errors:

```bash
pytest
ruff check .
```

## License

This project is licensed under the MIT License.
