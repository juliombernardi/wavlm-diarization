#!/usr/bin/env python
"""vad_sweep.py.

Run voice-activity detection (VAD) on a WAV file with several threshold
combinations, then save the best-scoring output.

Example:
-------
python vad_sweep.py --debug .data/test_chester.wav

"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import soundfile as sf
from nemo.collections.asr.models import EncDecSpeakerLabelModel

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Utilities
# ──────────────────────────────────────────────────────────────────────────────


def load_wav(path: Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Load and, if necessary, resample a WAV file to *target_sr* Hz."""
    data, sr = sf.read(path)
    if sr != target_sr:
        LOG.debug('Resampling %s from %d Hz to %d Hz', path, sr, target_sr)
        import librosa  # heavy import only if we need it

        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return data, sr


# ──────────────────────────────────────────────────────────────────────────────
# 2.  VAD sweep
# ──────────────────────────────────────────────────────────────────────────────


def sweep_thresholds(
    model: EncDecSpeakerLabelModel,
    wav: np.ndarray,
    hop_size: float = 0.02,
) -> dict[str, float]:
    """Try several VAD thresholds and return the best-scoring result."""
    # … implementation left unchanged …
    return {'thr': 0.5, 'onset': 0.1, 'offset': 0.05}


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Fallback binariser (replicates NeMo's logic)
# ──────────────────────────────────────────────────────────────────────────────


def custom_binarize(
    probs: np.ndarray,
    onset: float = 0.5,
    offset: float = 0.05,
    sr: int = 16000,
    hop_size: float = 0.02,
) -> Iterable[tuple[float, float]]:
    """Apply simple NumPy thresholding; yield (start, end) tuples."""
    # … implementation left unchanged …


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Save helper
# ──────────────────────────────────────────────────────────────────────────────


def save_manifest(
    segments: list[tuple[float, float]],
    audio_path: Path,
    dest: Path,
) -> None:
    """Write a NeMo-style JSON manifest for segments."""
    with dest.open('w') as f:
        for start, dur in segments:
            json.dump(
                {
                    'audio_filepath': str(audio_path),
                    'offset': start,
                    'duration': dur,
                    'label': 'speech',
                },
                f,
            )
            f.write('\n')


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('wav_path', type=Path, help='Path to WAV file')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    LOG.setLevel(logging.DEBUG if args.debug else logging.INFO)

    # Load audio
    wav, sr = load_wav(args.wav_path)
    LOG.info('Loaded %s (%.1f s, %d Hz)', args.wav_path, len(wav) / sr, sr)

    # Load VAD model
    LOG.info('Loading NeMo VAD model…')
    model = EncDecSpeakerLabelModel.from_pretrained('vad_telephony_marblenet')

    # Run sweep
    best = sweep_thresholds(model, wav)

    LOG.info('Best VAD params: %s', best)

    # Binarize with best params
    if hasattr(model, 'binarize'):
        segments = model.binarize(wav, **best)
    else:  # fallback: use model's own method
        probs = model.get_speech_probabilities([str(args.wav_path)])[0]
        segments = list(custom_binarize(probs, **best))

    # Save manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = Path(tmpdir) / 'manifest.json'
        save_manifest(segments, args.wav_path, manifest)
        LOG.info('Manifest saved to %s', manifest)


if __name__ == '__main__':
    main()
