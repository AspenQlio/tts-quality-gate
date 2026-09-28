# TTS Quality Gate

Un arnés automatizado de control de calidad para modelos de Text-to-Speech (TTS). Evita que los modelos degradados lleguen a producción después de un fine-tuning.

Este proyecto implementa integración continua para machine learning de audio. Transcribe el audio generado usando un transcriptor externo (como Whisper) y calcula la Tasa de Error de Palabras (WER) contra el texto de referencia. El pipeline falla si el WER agregado supera un límite definido.

## Arquitectura

El arnés evalúa un conjunto de datos en formato JSONL.

1. **Generación:** El modelo TTS genera un archivo de audio para cada texto de referencia.
2. **Transcripción:** Un transcriptor convierte el audio generado de nuevo a texto.
3. **Evaluación:** El pipeline normaliza ambos textos (minúsculas, sin puntuación) y calcula el WER.
4. **Decisión:** El pipeline agrega el WER y aplica el límite. Pasa solo si cada caso se mantiene por debajo del límite.

## Uso

Define un conjunto de evaluación (`dataset.jsonl`):

```json
{"id": "test-01", "reference_text": "The server responded with an error.", "audio_path": "outputs/test-01.wav"}
{"id": "test-02", "reference_text": "Restart the database cluster.", "audio_path": "outputs/test-02.wav"}
```

Ejecuta el control de calidad:

```bash
tts-quality-gate --dataset dataset.jsonl --threshold 0.05 --backend mock --json report.json
```

## Desarrollo

```bash
uv sync --all-extras --dev
pytest
ruff check .
```

La suite de pruebas verifica coincidencias exactas, eliminación de puntuación, invariancia de mayúsculas y la aplicación estricta del límite.

## Licencia

MIT
