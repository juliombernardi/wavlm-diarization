#!/usr/bin/env python3
"""
Run a grid-search over NeMo VAD options and aggregate results.

Usage:
    python vad_sweep.py path/to/my_audio.wav
"""

import argparse
import itertools
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from nemo.collections.asr.models import EncDecClassificationModel

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Try to import NeMo helpers; if they're not there we will provide our own
# ──────────────────────────────────────────────────────────────────────────────
try:
    from nemo.collections.asr.parts.utils.vad_utils import (
        get_speech_probabilities as nemo_get_probs,
        binarize_speech_probabilities as nemo_binarize,
    )
except ImportError:  # very old NeMo build
    nemo_get_probs = None
    nemo_binarize = None

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Hyper-parameter grid
# ──────────────────────────────────────────────────────────────────────────────
CHECKPOINTS = {
    "vad_telephony_marblenet": 8000,                 # 8 kHz narrow-band
    "Frame_VAD_Multilingual_MarbleNet_v2.0": 16000,  # 16 kHz wide-band
}

THRESHOLDS   = [0.5, 0.8]
ON_OFF_PAIRS = [(0.10, 0.05), (0.30, 0.02)]          # (min_on, min_off) seconds
SAMPLE_RATES = [8000, 16000]


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Fallback binariser (replicates NeMo’s logic)
# ──────────────────────────────────────────────────────────────────────────────
def custom_binarize(
    probs: np.ndarray,
    threshold: float,
    min_on: float,
    min_off: float,
    frame_hop: float,
):
    """
    Convert frame-level probabilities to [start_frame, end_frame) segments.

    Args
    ----
    probs      : 1-D numpy array of probabilities (0-1)
    threshold  : speech if p >= threshold
    min_on     : minimum speech segment length (sec)
    min_off    : minimum silence length that separates two speech segments (sec)
    frame_hop  : seconds between frames (0.02 for 20 ms)
    """
    speech = probs >= threshold
    segments = []
    idx = 0
    n_frames = len(probs)
    min_on_frames  = int(min_on  / frame_hop)
    min_off_frames = int(min_off / frame_hop)

    while idx < n_frames:
        # Skip non-speech frames
        if not speech[idx]:
            idx += 1
            continue

        # Found speech start
        seg_start = idx
        # Advance until silence longer than min_off
        silence_ctr = 0
        while idx < n_frames:
            idx += 1
            if idx == n_frames:
                break
            if speech[idx]:
                silence_ctr = 0
            else:
                silence_ctr += 1
                if silence_ctr >= min_off_frames:
                    break
        seg_end = idx - silence_ctr

        # Keep only if long enough
        if seg_end - seg_start >= min_on_frames:
            segments.append((seg_start, seg_end))

    return segments


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_audio(path, target_sr):
    """Load WAV and resample if needed; returns (mono_tensor, sr)."""
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav[0], target_sr


def run_vad(model_name, wav, sr, threshold, min_on, min_off):
    """Run VAD once and return summary stats."""

    # -- load model directly onto GPU ----------------------------------------
    model = (
        EncDecClassificationModel.from_pretrained(model_name, map_location="cuda")
        .cuda()
        .eval()
    )

    # -- get frame probabilities --------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        torchaudio.save(tmp.name, wav.unsqueeze(0), sr)

        if nemo_get_probs is not None:
            probs = nemo_get_probs(
                model,
                tmp.name,
                batch_size=64,
                window_size=0.02,
                hop_size=0.02,
            )
        else:  # fallback: use model’s own method
            probs = model.get_speech_probabilities(
                [tmp.name],
                batch_size=64,
                window_length_in_sec=0.02,
                shift_length_in_sec=0.02,
            )[0]

        os.remove(tmp.name)

    probs = np.asarray(probs)  # ensure numpy
    frame_hop = 0.02           # 20 ms

    # -- binarise ------------------------------------------------------------
    if nemo_binarize is not None:
        segments = nemo_binarize(
            probs,
            threshold=threshold,
            min_speech_duration_ms=int(min_on * 1000),
            min_silence_duration_ms=int(min_off * 1000),
            window_length_samples=int(sr * frame_hop),
        )
    else:
        segments = custom_binarize(
            probs, threshold, min_on, min_off, frame_hop
        )

    # Convert segments to boolean mask for stats
    speech_mask = np.zeros_like(probs, dtype=bool)
    for beg, end in segments:
        speech_mask[beg:end] = True

    pct_speech = 100 * speech_mask.mean()
    speech_sec = speech_mask.sum() * frame_hop

    return {
        "model": model_name,
        "audio_sr": sr,
        "model_sr": CHECKPOINTS[model_name],
        "threshold": threshold,
        "min_on": min_on,
        "min_off": min_off,
        "speech_%": round(pct_speech, 2),
        "speech_sec": round(speech_sec, 2),
        "n_segments": len(segments),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", type=Path, help="Path to WAV file")
    args = parser.parse_args()

    results = []

    for sr, model_name, threshold, (min_on, min_off) in itertools.product(
        SAMPLE_RATES, CHECKPOINTS, THRESHOLDS, ON_OFF_PAIRS
    ):
        wav, _ = load_audio(args.wav_path, sr)
        stats = run_vad(model_name, wav, sr, threshold, min_on, min_off)
        results.append(stats)

        print(
            f"✓ {model_name:<40} sr={sr:<5} thr={threshold:<4} "
            f"on/off={min_on}/{min_off} "
            f"→ speech {stats['speech_%']:.1f}% "
            f"({stats['speech_sec']} s, {stats['n_segments']} segs)"
        )

    df = pd.DataFrame(results)
    df.to_csv("vad_results.csv", index=False)
    print("\nSummary saved to vad_results.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
