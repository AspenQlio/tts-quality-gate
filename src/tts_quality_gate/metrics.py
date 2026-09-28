from __future__ import annotations

import jiwer


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculates the Word Error Rate between a reference and a hypothesis.

    Returns 0.0 for a perfect match. Normalizes text by lowercasing and
    stripping punctuation before calculation to prevent artificial penalties.
    """
    transform = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )

    # Handle edge case: empty reference
    if not reference.strip():
        return 1.0 if hypothesis.strip() else 0.0

    return float(
        jiwer.wer(
            reference,
            hypothesis,
            reference_transform=transform,
            hypothesis_transform=transform,
        )
    )
