# Music-ML (Bayen Lab) — Piano Performance → Engraved Score

A research project on turning a piano **performance** (audio or MIDI) into a correct **engraved score** (MusicXML / sheet music), as accurately as possible — the *transcription/notation* half of the lab's larger goal (natural language → score).

> **📄 Start here:** [`LAB_REPORT_COMPLETE.md`](LAB_REPORT_COMPLETE.md) — the complete, canonical lab report (every component, the design choices, the research finding, and how it fits the lab's direction).
> **🎤 Presentation:** [`LAB_PRESENTATION_SCRIPT.md`](LAB_PRESENTATION_SCRIPT.md) — the joint lab-talk script (this work + the generative side).

---

## Headline results

- **Reproduced the published SOTA to within noise** — MeanER **11.18** vs the paper's **11.30** (Beyer & Dai, *MIDI2ScoreTransformer*, ISMIR 2024) on the 59-performance ASAP test set. This validates the whole eval harness, so every other number is measured against a real reproduction. → [`benchmark/BASELINE_RESULTS.md`](benchmark/BASELINE_RESULTS.md)
- **On clean, in-distribution piano the full audio→score pipeline scores ~2.75 MeanER** — i.e. it **matches commercial quality on easy repertoire**, which is expected because the Block-2 score model is the same open released checkpoint that **Songscription** is built on. (2.75 is the easy end of the difficulty range; the apples-to-apples number is the 11.18 reproduction.) → [`docs/reports/SONGSCRIPTION_PARITY.md`](docs/reports/SONGSCRIPTION_PARITY.md)
- **The accuracy ceiling is rhythm, not the transcriber and not pitch.** Stage decomposition shows feeding *perfect* MIDI into Block 2 fails the same way as full audio→score (Block 1 is ~95–97% correct). The notes are right; **triplet/tuplet onset placement** is where it breaks on dense Romantic music. → [`benchmark/DECOMPOSED_FINDINGS.md`](benchmark/DECOMPOSED_FINDINGS.md)
- **Core research finding (carefully stated):** across a broad matrix of data, representation, loss, and training schemes, **no lever achieved correct-*rate* triplet placement** (corpus reshaping *over*-shot to 19.9%; everything else under-shot or collapsed). We matched the released model's validation loss and came within ~1 MUSTER point, but never its correct-rate placement. **The leading, best-supported explanation is the training data** (our corpus is ~1.7% tuplets); the decisive converged-on-tuplet-rich-data run was left undertrained, so this is *"not recovered within our compute/data budget,"* not a proof of impossibility. → [`LAB_REPORT_COMPLETE.md` §7](LAB_REPORT_COMPLETE.md)
- **Strategic takeaway:** use the released model as the **placement-capable engine** and as a **data engine** (transcription manufactures score corpora to pretrain the generative side); own the shared **music-notation representation**. → [`LAB_REPORT_COMPLETE.md` §9](LAB_REPORT_COMPLETE.md)

---

## The system: two blocks

```
 AUDIO ──[ Block 1: transcription ]──► performance MIDI ──[ Block 2: notation ]──► MusicXML ──► PDF
        Basic Pitch / hFT / MT3 / Transkun        MIDI2ScoreTransformer (Beyer & Dai)     MuseScore
```

- **Block 1 (audio → MIDI)** is solved-enough: off-the-shelf transcribers recover ~95–97% of notes. Best: **hFT-Transformer** (Sony, ISMIR 2023, 96.72% onset-F1). We proved this is **not** the bottleneck and deliberately stopped optimizing it.
- **Block 2 (MIDI → score)** is the hard, interesting problem and where the whole accuracy ceiling lives — see the research finding above.

Full eval harness, the tuplet-placement investigation, and all design choices are documented in [`LAB_REPORT_COMPLETE.md`](LAB_REPORT_COMPLETE.md).

---

## Repository map

```
LAB_REPORT_COMPLETE.md      ← canonical lab report (read this first)
LAB_PRESENTATION_SCRIPT.md  ← joint lab-presentation script
transcribe.py               ← pipeline CLI entry point
MIDI2ScoreTransformer/      ← Block 2 model + our training stack (train.py is ours; upstream ships inference only)
hFT-Transformer/ , mt3/     ← Block 1 transcribers (checkpoints downloaded separately)
benchmark/                  ← eval harness, MUSTER results, test pieces (BASELINE/DECOMPOSED/TRACKB results, LEAKAGE_AUDIT)
scripts/                    ← data-engine builders, B2 tokenizer, diagnostics, MUSTER rerank
docs/
  reports/                  ← topic reports & roadmaps (SONGSCRIPTION_PARITY, AUDIO_SCORE_ROADMAP, DATA_SCALING_DERISK, …)
  sessions/                 ← dated session logs / morning reports (provenance of the autonomous runs)
  archive/                  ← raw dumps & intermediate JSONs (kept for reference, not canonical)
```

---

## Setup & usage

### 1. Python environment (Python 3.11)
```bash
python3.11 -m venv venv311 && source venv311/bin/activate
pip install -r requirements.txt
pip install transkun torchcodec
# Pins that matter: transformers==4.44.2, music21 9.x. MT3 needs a separate venv (JAX/TF) — see mt3_inference.py.
```

### 2. Model checkpoints (downloaded separately; gitignored)
- **MIDI2ScoreTransformer** (`MIDI2ScoreTF.ckpt`): [releases](https://github.com/TimFelixBeyer/MIDI2ScoreTransformer/releases) → `MIDI2ScoreTransformer/checkpoints/`
- **hFT-Transformer**: [ismir2023 release](https://github.com/sony/hFT-Transformer/releases/tag/ismir2023) → `hFT-Transformer/checkpoint/`
- **MT3** (optional): `gs://mt3/checkpoints/ismir2021/` → `mt3/checkpoints/ismir2021/`

### 3. System deps
`brew install lilypond` (music21 backend) · `brew install --cask musescore` (musescore/transformer backends).

### Run it
```bash
# Best pipeline: hFT transcriber + neural (MIDI2ScoreTransformer) backend
python transcribe.py samples/TwinkleTwinkle.mp3 -t hft -b transformer

# Skip transcription, score an existing MIDI directly
python transcribe.py --midi-input midi/existing.mid -b transformer
```
Output defaults to `outputs/<backend>/<input_stem>.pdf`.

- `-t / --transcriber`: `basic-pitch` (default) · `mt3` (needs `venv_mt3`) · `transkun` · **`hft`** (best)
- `-b / --backend`: `music21` (default) · `musescore` · **`transformer`** (neural, 2-hand notation)

### Evaluation
```bash
python benchmark/eval_tier1_asap.py     # paper-comparable MUSTER on the 59-perf ASAP test (CPU on macOS)
python benchmark/eval_padsweep.py       # fast single-pass pad-threshold sweep (the "standing" measurement)
```

---

## Key references
- Beyer & Dai, *End-to-end Piano Performance-MIDI to Score Conversion with Transformers*, **ISMIR 2024** — [arXiv:2410.00210](https://arxiv.org/abs/2410.00210) · [code](https://github.com/TimFelixBeyer/MIDI2ScoreTransformer)
- hFT-Transformer (Sony, ISMIR 2023) · Basic Pitch (Spotify) · MT3 (Google Magenta, ISMIR 2021) · Transkun
- **Songscription** — the commercial audio→sheet-music product built on the Beyer & Dai architecture (co-founder Tim Beyer = the paper's lead author).
