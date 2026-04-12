#!/usr/bin/env python3
"""Piano audio → sheet music transcription pipeline.

Chains: Audio (wav/mp3) → [transcriber] → MIDI → [backend] → PDF/PNG

Transcribers:
  basic-pitch — Spotify Basic Pitch CNN (default)
  mt3         — Google MT3 T5 Transformer (requires venv_mt3)

Backends:
  music21     — music21 heuristics + LilyPond rendering (default)
  musescore   — MuseScore CLI MIDI import + rendering
  transformer — MIDI2ScoreTransformer (neural) + MuseScore rendering
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger("transcribe")

SUPPORTED_AUDIO = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
SUPPORTED_OUTPUT = {".pdf", ".png"}
TRANSCRIBERS = ["basic-pitch", "mt3"]
BACKENDS = ["music21", "musescore", "transformer"]

SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFORMER_DIR = SCRIPT_DIR / "MIDI2ScoreTransformer"
CHECKPOINT_PATH = TRANSFORMER_DIR / "checkpoints" / "MIDI2ScoreTF.ckpt"


# ---------------------------------------------------------------------------
# Phase 1: Audio → MIDI
# ---------------------------------------------------------------------------

def audio_to_midi_basic_pitch(audio_path: Path, midi_path: Path) -> int:
    """Run Basic Pitch inference and write a MIDI file."""
    log.info("[1/3] Audio → MIDI (Basic Pitch)")
    log.info("      Input:  %s", audio_path)

    _root = logging.getLogger()
    _prev_level = _root.level
    _root.setLevel(logging.ERROR)

    import io, contextlib
    with contextlib.redirect_stderr(io.StringIO()), \
         contextlib.redirect_stdout(io.StringIO()):
        from basic_pitch.inference import predict
        model_output, midi_data, note_events = predict(str(audio_path))

    _root.setLevel(_prev_level)
    midi_data.write(str(midi_path))

    n_notes = len(note_events)
    log.info("      Notes detected: %d", n_notes)
    log.info("      MIDI saved: %s", midi_path)
    return n_notes


def audio_to_midi_mt3(audio_path: Path, midi_path: Path) -> int:
    """Run MT3 Transformer inference and write a MIDI file."""
    log.info("[1/3] Audio → MIDI (MT3)")
    log.info("      Input:  %s", audio_path)

    from mt3_inference import transcribe_audio
    n_notes = transcribe_audio(str(audio_path), str(midi_path))

    log.info("      Notes detected: %d", n_notes)
    log.info("      MIDI saved: %s", midi_path)
    return n_notes


def audio_to_midi(audio_path: Path, midi_path: Path, transcriber: str = "basic-pitch") -> int:
    """Dispatch to the chosen transcriber."""
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    if transcriber == "basic-pitch":
        return audio_to_midi_basic_pitch(audio_path, midi_path)
    elif transcriber == "mt3":
        return audio_to_midi_mt3(audio_path, midi_path)
    else:
        raise ValueError(f"Unknown transcriber: {transcriber}")


# ---------------------------------------------------------------------------
# Phase 2 + 3 helpers
# ---------------------------------------------------------------------------

def find_lilypond() -> Optional[str]:
    """Locate the LilyPond binary."""
    path = shutil.which("lilypond")
    if path:
        return path
    for candidate in ["/opt/homebrew/bin/lilypond", "/usr/local/bin/lilypond"]:
        if Path(candidate).is_file():
            return candidate
    return None


def find_musescore() -> Optional[str]:
    """Locate the MuseScore binary."""
    path = shutil.which("mscore")
    if path:
        return path
    for candidate in [
        "/opt/homebrew/bin/mscore",
        "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
        "/usr/local/bin/mscore",
    ]:
        if Path(candidate).is_file():
            return candidate
    return None


def run_musescore(mscore_path: str, input_path: str, output_path: str) -> None:
    """Run MuseScore in converter mode."""
    env = os.environ.copy()
    if sys.platform == "linux":
        env["QT_QPA_PLATFORM"] = "offscreen"
    subprocess.run(
        [mscore_path, "-o", output_path, input_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        check=True,
    )


# ---------------------------------------------------------------------------
# Backend: music21 (heuristics + LilyPond)
# ---------------------------------------------------------------------------

def backend_music21(midi_path: Path, output_path: Path) -> Path:
    from music21 import converter, environment, note, chord

    log.info("[2/3] MIDI → Score (music21)")
    score = converter.parse(str(midi_path))

    key = score.analyze("key")
    log.info("      Key: %s", key)

    time_sigs = score.recurse().getElementsByClass("TimeSignature")
    ts_str = time_sigs[0].ratioString if time_sigs else "not detected"
    log.info("      Time signature: %s", ts_str)

    parts = score.parts
    log.info("      Parts: %d", len(parts))

    notes = score.recurse().getElementsByClass((note.Note, chord.Chord))
    log.info("      Notes/chords: %d", len(notes))

    tempos = score.recurse().getElementsByClass("MetronomeMark")
    if tempos:
        t = tempos[0]
        log.info("      Tempo: %s BPM", int(t.number) if t.number else t)

    log.info("[3/3] Score → Sheet Music (LilyPond)")
    ext = output_path.suffix.lower()
    lily_fmt = "lily.pdf" if ext == ".pdf" else "lily.png"

    lilypond_path = find_lilypond()
    if lilypond_path:
        log.info("      LilyPond: %s", lilypond_path)
        env = environment.Environment()
        env["lilypondPath"] = lilypond_path
        try:
            stem = str(output_path.with_suffix(""))
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            saved_stderr = os.dup(2)
            os.dup2(devnull_fd, 2)
            try:
                written = Path(str(score.write(lily_fmt, fp=stem)))
            finally:
                os.dup2(saved_stderr, 2)
                os.close(devnull_fd)
                os.close(saved_stderr)
            if written != output_path and written.exists():
                written.rename(output_path)
                written = output_path
            ly_source = Path(stem)
            if ly_source.exists() and ly_source.suffix != output_path.suffix:
                ly_source.unlink()
            log.info("      Output: %s", written)
            return written
        except Exception as exc:
            log.warning("      LilyPond rendering failed: %s", exc)
            log.warning("      Falling back to MusicXML export.")
    else:
        log.warning("      LilyPond not found. Falling back to MusicXML export.")

    xml_path = output_path.with_suffix(".musicxml")
    written = Path(str(score.write("musicxml", fp=str(xml_path))))
    log.info("      Output (MusicXML): %s", written)
    return written


# ---------------------------------------------------------------------------
# Backend: MuseScore CLI
# ---------------------------------------------------------------------------

def backend_musescore(midi_path: Path, output_path: Path) -> Path:
    mscore = find_musescore()
    if not mscore:
        log.error("MuseScore not found. Install via: brew install --cask musescore")
        sys.exit(1)

    log.info("[2/3] MIDI → Score + Render (MuseScore)")
    log.info("      MuseScore: %s", mscore)

    run_musescore(mscore, str(midi_path), str(output_path))

    if output_path.exists():
        log.info("      Output: %s", output_path)
        return output_path
    else:
        log.error("      MuseScore failed to produce output.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Backend: MIDI2ScoreTransformer
# ---------------------------------------------------------------------------

def backend_transformer(midi_path: Path, output_path: Path) -> Path:
    if not TRANSFORMER_DIR.is_dir():
        log.error("MIDI2ScoreTransformer repo not found at %s", TRANSFORMER_DIR)
        log.error("Clone it: git clone https://github.com/TimFelixBeyer/MIDI2ScoreTransformer.git")
        sys.exit(1)
    if not CHECKPOINT_PATH.is_file():
        log.error("Model checkpoint not found at %s", CHECKPOINT_PATH)
        log.error("Download from: https://github.com/TimFelixBeyer/MIDI2ScoreTransformer/releases")
        sys.exit(1)

    mscore = find_musescore()
    if not mscore:
        log.error("MuseScore not found (required for transformer rendering).")
        log.error("Install via: brew install --cask musescore")
        sys.exit(1)

    log.info("[2/3] MIDI → Score (MIDI2ScoreTransformer)")

    # Add the transformer source to Python path
    transformer_src = str(TRANSFORMER_DIR / "midi2scoretransformer")
    if transformer_src not in sys.path:
        sys.path.insert(0, transformer_src)

    import torch
    from config import MyModelConfig
    # Patch for newer transformers compatibility
    if not hasattr(MyModelConfig, '_attn_implementation_internal'):
        MyModelConfig._attn_implementation_internal = None
    torch.serialization.add_safe_globals([MyModelConfig])

    from tokenizer import MultistreamTokenizer
    from utils import infer
    from score_utils import postprocess_score
    from models.roformer import Roformer

    # MPS produces numerically incorrect pad logits in this model's attention
    # layers, causing every note to be masked out. Force CPU for correctness.
    device = "cpu"
    log.info("      Device: %s", device)
    log.info("      Loading model...")

    model = Roformer.load_from_checkpoint(
        str(CHECKPOINT_PATH), map_location=device, weights_only=False,
    )
    model.eval()
    model.to(device)

    log.info("      Tokenizing MIDI...")
    x = MultistreamTokenizer.tokenize_midi(str(midi_path))

    log.info("      Running inference (%d notes)...", x["pitch"].shape[0])
    y_hat = infer(x, model, verbose=False, kv_cache=True)

    log.info("      Decoding to score...")
    mxl = MultistreamTokenizer.detokenize_mxl(y_hat)
    mxl = postprocess_score(mxl, inPlace=True)

    # Write MusicXML, then render to PDF/PNG with MuseScore
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp:
        tmp_mxl = tmp.name
    mxl.write("musicxml", fp=tmp_mxl)

    log.info("[3/3] Score → Sheet Music (MuseScore)")
    log.info("      MuseScore: %s", mscore)
    run_musescore(mscore, tmp_mxl, str(output_path))
    os.unlink(tmp_mxl)

    if output_path.exists():
        log.info("      Output: %s", output_path)
        return output_path
    else:
        log.error("      MuseScore rendering failed.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe piano audio to sheet music.",
        epilog="Example: python transcribe.py input.wav -o score.pdf --backend musescore",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input audio file (wav, mp3, ogg, flac, m4a)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file path (default: outputs/<backend>/<input_stem>.pdf). Extension sets format.",
    )
    parser.add_argument(
        "-m", "--keep-midi",
        type=Path,
        default=None,
        help="Path for intermediate MIDI file (default: <input_stem>.mid beside output)",
    )
    parser.add_argument(
        "-t", "--transcriber",
        choices=TRANSCRIBERS,
        default="basic-pitch",
        help="Audio-to-MIDI transcriber: basic-pitch (Spotify CNN) or "
             "mt3 (Google T5 Transformer, requires venv_mt3). Default: basic-pitch.",
    )
    parser.add_argument(
        "-b", "--backend",
        choices=BACKENDS,
        default="music21",
        help="Score conversion backend: music21 (heuristics + LilyPond), "
             "musescore (MuseScore MIDI import), "
             "transformer (MIDI2ScoreTransformer neural model). Default: music21.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import warnings
    warnings.filterwarnings("ignore")

    audio_path: Path = args.input
    if not audio_path.is_file():
        log.error("Input file not found: %s", audio_path)
        sys.exit(1)
    if audio_path.suffix.lower() not in SUPPORTED_AUDIO:
        log.error(
            "Unsupported audio format '%s'. Supported: %s",
            audio_path.suffix, ", ".join(sorted(SUPPORTED_AUDIO)),
        )
        sys.exit(1)

    backend = args.backend
    output_path: Path = args.output or (
        SCRIPT_DIR / "outputs" / backend / (audio_path.stem + ".pdf")
    )
    if output_path.suffix.lower() not in SUPPORTED_OUTPUT:
        log.error(
            "Unsupported output format '%s'. Supported: %s",
            output_path.suffix, ", ".join(sorted(SUPPORTED_OUTPUT)),
        )
        sys.exit(1)

    midi_path: Path = args.keep_midi or output_path.with_name(audio_path.stem + ".mid")

    # Phase 1: Audio → MIDI
    transcriber = args.transcriber
    try:
        audio_to_midi(audio_path, midi_path, transcriber=transcriber)
    except Exception as exc:
        log.error("Audio → MIDI failed: %s", exc)
        sys.exit(1)

    print()

    # Phase 2 + 3: MIDI → Score → Render (backend-dependent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Backend: %s", backend)
    print()

    try:
        if backend == "music21":
            backend_music21(midi_path, output_path)
        elif backend == "musescore":
            backend_musescore(midi_path, output_path)
        elif backend == "transformer":
            backend_transformer(midi_path, output_path)
    except Exception as exc:
        log.error("Pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
