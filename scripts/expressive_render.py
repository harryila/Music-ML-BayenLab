"""Render a MusicXML score to a perturbed performance MIDI plus an alignment json.

The synthetic MIDI is meant to look like a (loose) human performance: tempo
drifts, velocity wiggles, notes start a bit late or early, durations vary.
The corresponding MusicXML score is left untouched. Together this gives us a
synthetic (perturbed-MIDI, MusicXML) training pair for MIDI2ScoreTransformer.

Crucially, alignment is exact by construction: notes are extracted from the
score using the SAME mxl_to_list canonical order MIDI2ScoreTransformer uses,
and written to MIDI in that order. midi_to_list (which sorts by (start, pitch,
duration)) produces the same ordering because we apply per-chord onset jitter
(notes sharing a score offset share a jittered onset), so chord-internal order
is preserved.

Robustness:
    The whole `mxl_to_list(...)` call (and downstream processing) is wrapped
    in try/except. If parsing fails (PDMX has ~42 corrupted files plus rarer
    parser errors), this returns {ok: False, error: ...}. The caller
    (make_pairs.py) is expected to log + skip cleanly.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pretty_midi

# Set up the import path so we can use MIDI2ScoreTransformer's tokenizer.
SCRIPT_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = SCRIPT_DIR / "MIDI2ScoreTransformer" / "midi2scoretransformer"
if str(TOKENIZER_DIR) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_DIR))

log = logging.getLogger(__name__)


@dataclass
class Perturbations:
    """Per-piece perturbation knobs. All randomized via the seed."""
    tempo_walk_std: float = 0.10        # ~N(1.0, 0.10) per-bar multiplier
    tempo_smooth_window: int = 3         # bars of moving average over the walk
    velocity_offset_std: float = 10.0    # per-note velocity ~N(0, 10)
    phrase_loudness_amp: int = 15        # phrase-level cosine envelope ±15
    phrase_length_bars: int = 8
    onset_jitter_seconds: float = 0.020  # ±20 ms gaussian (per chord, not per note)
    duration_scale_std: float = 0.05     # multiplicative N(1, 0.05)
    early_release_prob: float = 0.05
    early_release_factor: float = 0.7


def _measure_starts_ql(score) -> list[float]:
    """Sorted unique measure-start offsets in quarter-lengths."""
    starts = set()
    for part in score.parts:
        for m in part.getElementsByClass("Measure"):
            starts.add(float(m.offset))
    if not starts:
        return [0.0]
    return sorted(starts)


def _build_q_to_s(score, perturb: Perturbations, rng: random.Random,
                  max_ql: float) -> callable:
    """Build a quarter-length-to-seconds mapping with a smooth tempo walk per bar."""
    flat = score.flatten()
    base_bpm = 100.0
    for tempo in flat.getElementsByClass("MetronomeMark"):
        try:
            n = float(tempo.number)
            if n > 0:
                base_bpm = n
                break
        except Exception:
            continue

    bar_starts = _measure_starts_ql(score)
    bar_starts = [b for b in bar_starts if b <= max_ql + 0.001]
    if not bar_starts or bar_starts[0] > 0:
        bar_starts = [0.0] + bar_starts
    bar_starts.append(max_ql + 1.0)

    n_bars = len(bar_starts) - 1
    walk = np.array([rng.gauss(1.0, perturb.tempo_walk_std) for _ in range(n_bars)])
    if perturb.tempo_smooth_window > 1:
        kernel = np.ones(perturb.tempo_smooth_window) / perturb.tempo_smooth_window
        walk = np.convolve(walk, kernel, mode="same")
    walk = np.clip(walk, 0.5, 2.0)

    def q_to_s(q: float) -> float:
        # Find which bar this offset falls in
        idx = 0
        for i in range(n_bars):
            if bar_starts[i] <= q < bar_starts[i + 1]:
                idx = i
                break
        else:
            idx = n_bars - 1
        seconds = 0.0
        for i in range(idx):
            bar_q = bar_starts[i + 1] - bar_starts[i]
            local_bpm = base_bpm * walk[i]
            seconds += bar_q * 60.0 / local_bpm
        local_bpm = base_bpm * walk[idx]
        seconds += (q - bar_starts[idx]) * 60.0 / local_bpm
        return seconds

    return q_to_s


def render(mxl_path: Path, out_midi_path: Path, seed: int = 0,
           perturb: Perturbations | None = None) -> dict:
    """Render a MusicXML score to a perturbed performance MIDI.

    Returns:
        {ok: True, n_notes: int, alignment_path: str, n_measures: int}
        {ok: False, error: str}
    """
    if perturb is None:
        perturb = Perturbations()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from tokenizer import MultistreamTokenizer
            notes_list, score = MultistreamTokenizer.mxl_to_list(str(mxl_path))
    except Exception as exc:
        return {"ok": False, "error": f"parse: {type(exc).__name__}: {exc}"}

    if not notes_list:
        return {"ok": False, "error": "no_notes_in_score"}

    try:
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        # Source-score offsets and pitches in mxl_to_list canonical order
        score_offsets = [float(n.offset) for n in notes_list]
        score_durations = [float(n.duration.quarterLength) for n in notes_list]
        score_pitches = [int(n.pitch.midi) for n in notes_list]
        # source velocity (already realized by mxl_to_list)
        score_velocities = [int(n.volume.velocity or 80) for n in notes_list]

        max_ql = max(o + d for o, d in zip(score_offsets, score_durations))
        q_to_s = _build_q_to_s(score, perturb, rng, max_ql)

        # Per-chord (per-offset) onset jitter so chord-internal order is preserved
        unique_offsets = sorted(set(score_offsets))
        onset_jitters = {
            o: float(np_rng.normal(0, perturb.onset_jitter_seconds))
            for o in unique_offsets
        }

        # Phrase-level loudness envelope
        def phrase_loudness(t_ql: float) -> int:
            phase = (t_ql / (perturb.phrase_length_bars * 4.0)) * 2 * math.pi
            return int(perturb.phrase_loudness_amp * math.cos(phase))

        # Determine BEAT index for each score note (for chunks.json — matches ASAP
        # chunker.py which chunks per beat via midi_score_beats / performance_beats).
        # We approximate "beats" as quarter-note boundaries from offset 0.0 to max_ql.
        # This gives finer-grained crops in training (more random_crop options).
        n_beats = int(math.ceil(max_ql)) + 1

        def beat_idx(offset_ql: float) -> int:
            return min(n_beats - 1, int(offset_ql))

        # We still keep measure_idx around for diagnostics / future use.
        measure_starts = _measure_starts_ql(score)
        if measure_starts and measure_starts[0] > 0:
            measure_starts = [0.0] + measure_starts

        def measure_idx(offset: float) -> int:
            idx = 0
            for i, m_start in enumerate(measure_starts):
                if m_start <= offset:
                    idx = i
                else:
                    break
            return idx

        # Build MIDI in mxl_to_list order
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0, is_drum=False, name="Piano")
        alignment_entries = []
        for i, (o_ql, d_ql, p, v) in enumerate(
                zip(score_offsets, score_durations, score_pitches, score_velocities)):
            onset_s = q_to_s(o_ql) + onset_jitters[o_ql]
            offset_end_s = q_to_s(o_ql + d_ql) + onset_jitters[o_ql]
            dur_s = max(0.02, offset_end_s - onset_s)
            dur_s *= float(np_rng.normal(1.0, perturb.duration_scale_std))
            if rng.random() < perturb.early_release_prob:
                dur_s *= perturb.early_release_factor
            dur_s = max(0.02, dur_s)

            vel_offset = int(np_rng.normal(0, perturb.velocity_offset_std))
            new_vel = v + vel_offset + phrase_loudness(o_ql)
            new_vel = int(max(1, min(127, new_vel)))

            inst.notes.append(pretty_midi.Note(
                velocity=new_vel, pitch=int(p),
                start=float(onset_s), end=float(onset_s + dur_s),
            ))
            alignment_entries.append({
                "midi_idx": i,
                "score_idx": i,
                "pitch": int(p),
                "score_offset_ql": o_ql,
                "score_duration_ql": d_ql,
                "measure_idx": measure_idx(o_ql),
                "beat_idx": beat_idx(o_ql),
            })

        pm.instruments.append(inst)

        out_midi_path.parent.mkdir(parents=True, exist_ok=True)
        pm.write(str(out_midi_path))

        alignment_path = out_midi_path.with_suffix(".alignment.json")
        n_meas = max((a["measure_idx"] for a in alignment_entries), default=0) + 1
        n_beat = max((a["beat_idx"] for a in alignment_entries), default=0) + 1
        with alignment_path.open("w") as f:
            json.dump({
                "alignment": alignment_entries,
                "n_score_notes": len(notes_list),
                "n_midi_notes": len(notes_list),
                "n_measures": n_meas,
                "n_beats": n_beat,
                "max_ql": max_ql,
            }, f)

        return {
            "ok": True,
            "n_notes": len(notes_list),
            "alignment_path": str(alignment_path),
            "n_measures": n_meas,
            "n_beats": n_beat,
        }
    except Exception as exc:
        return {"ok": False, "error": f"render: {type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mxl", type=Path, help="Input .mxl/.musicxml")
    ap.add_argument("out", type=Path, help="Output .mid")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = render(args.mxl, args.out, seed=args.seed)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
