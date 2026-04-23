# Piano Audio → Sheet Music Pipeline

Transcribes piano audio recordings into rendered sheet music. The pipeline has
two configurable phases:

**Phase 1 — Audio to MIDI** (choose one transcriber):

- **basic-pitch** — [Spotify Basic Pitch](https://github.com/spotify/basic-pitch) CNN (default)
- **mt3** — [Google MT3](https://github.com/magenta/mt3) T5 Transformer (ISMIR 2021, requires venv_mt3)
- **transkun** — [Transkun](https://github.com/Yujia-Yan/Transkun) event-based, fully continuous timestamps
- **hft** — [hFT-Transformer](https://github.com/sony/hFT-Transformer) hierarchical Transformer (ISMIR 2023, sub-frame precision)

**Phase 2+3 — MIDI to Score to PDF** (choose one backend):

- **music21** — rule-based heuristics, rendered via [LilyPond](https://lilypond.org)
- **musescore** — [MuseScore](https://musescore.org) CLI MIDI import + rendering
- **transformer** — [MIDI2ScoreTransformer](https://github.com/TimFelixBeyer/MIDI2ScoreTransformer)
  neural model (ISMIR 2024), rendered via MuseScore

Best pipeline: **hFT → MIDI2ScoreTransformer** (both run in venv311, with direct
note handoff bypassing the MIDI file round-trip).

## Setup

### 1. Python environment

Requires Python 3.11 for the transformer backend, hFT, and Transkun.

```bash
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
pip install transkun torchcodec
```

MT3 requires a separate venv due to JAX/TensorFlow dependencies — see
`mt3_inference.py` and `mt3/` for setup instructions.

### 2. Model checkpoints

Download separately (not included in repo due to size):

- **MIDI2ScoreTransformer**: Download `MIDI2ScoreTF.ckpt` from
  [GitHub releases](https://github.com/TimFelixBeyer/MIDI2ScoreTransformer/releases)
  into `MIDI2ScoreTransformer/checkpoints/`
- **hFT-Transformer**: Download `checkpoint.zip` from
  [GitHub releases](https://github.com/sony/hFT-Transformer/releases/tag/ismir2023),
  unzip into `hFT-Transformer/checkpoint/`
- **MT3**: Download from `gs://mt3/checkpoints/ismir2021/` into `mt3/checkpoints/ismir2021/`

### 3. System dependencies

**LilyPond** (for music21 backend):
```bash
brew install lilypond          # macOS
sudo apt-get install lilypond  # Linux
```

**MuseScore** (for musescore and transformer backends):
```bash
brew install --cask musescore  # macOS
```

## Usage

```bash
# Default: basic-pitch transcriber, music21 backend
python transcribe.py samples/TwinkleTwinkle.mp3

# Best pipeline: hFT transcriber + transformer backend
python transcribe.py samples/TwinkleTwinkle.mp3 -t hft -b transformer

# Other transcribers
python transcribe.py samples/TwinkleTwinkle.mp3 -t transkun -b musescore
python transcribe.py samples/TwinkleTwinkle.mp3 -t mt3 -b transformer  # requires venv_mt3

# Skip Phase 1 — use a pre-existing MIDI file directly
python transcribe.py --midi-input midi/existing.mid -b transformer

# Tune Basic Pitch thresholds (higher = fewer false detections)
python transcribe.py samples/TwinkleTwinkle.mp3 -t basic-pitch --onset-threshold 0.6 --frame-threshold 0.4
```

Output defaults to `outputs/<backend>/<input_stem>.pdf` when `-o` is not given.

### Arguments

| Argument | Description |
|---|---|
| `input` | Path to audio file (.wav, .mp3, .ogg, .flac, .m4a). Optional if `--midi-input` is used. |
| `--midi-input` | Skip Phase 1 and use this MIDI file directly for Phase 2+3. |
| `-o`, `--output` | Output file path. Extension sets format (.pdf or .png). Default: `outputs/<backend>/<input_stem>.pdf` |
| `-m`, `--keep-midi` | Path for intermediate MIDI file. Default: `<input_stem>.mid` beside output |
| `-t`, `--transcriber` | Audio-to-MIDI transcriber: `basic-pitch` (default), `mt3`, `transkun`, or `hft` |
| `-b`, `--backend` | Score backend: `music21` (default), `musescore`, or `transformer` |
| `--onset-threshold` | Basic Pitch onset threshold (0-1). Higher = fewer false onsets. Default: 0.5 |
| `--frame-threshold` | Basic Pitch frame threshold (0-1). Higher = fewer phantom notes. Default: 0.3 |

### Transcribers

| | basic-pitch | mt3 | transkun | hft |
|---|---|---|---|---|
| Architecture | CNN | T5 Transformer | Event-based | Hierarchical Transformer |
| Timing | Frame-level | 10ms grid | Continuous | Sub-frame (~1-2ms) |
| Training data | General audio | MAESTRO | MAESTRO | MAESTRO V3 |
| Onset F1 | — | ~96% | — | 96.72% |
| Venv | any | venv_mt3 | venv311 | venv311 |

### Backends

| | music21 | musescore | transformer |
|---|---|---|---|
| Approach | Rule-based heuristics | MuseScore MIDI import | Neural seq2seq model |
| Hand splitting | None (single part) | None (single part) | Learned (2 parts) |
| Time sig | Inferred | MuseScore internal | Predicted (with majority-vote correction) |
| Renderer | LilyPond | MuseScore | MuseScore |
| System deps | LilyPond | MuseScore | MuseScore + PyTorch |

## Project Structure

```
transcribe.py                CLI entry point — full pipeline
mt3_inference.py             MT3 inference wrapper (JAX/T5X)
samples/                     Input audio files
midi/                        Intermediate MIDI files (gitignored, regenerable)
outputs/                     Output PDFs/PNGs (gitignored, regenerable)
MIDI2ScoreTransformer/       MIDI-to-score neural model (ISMIR 2024)
  checkpoints/               Pre-trained weights (gitignored, download separately)
  midi2scoretransformer/     Tokenizer, model, inference, score post-processing
hFT-Transformer/             Audio-to-MIDI transcriber (ISMIR 2023)
  checkpoint/                Pre-trained weights (gitignored, download separately)
  model/                     AMT class, model architecture
mt3/                         Audio-to-MIDI transcriber (ISMIR 2021)
  checkpoints/               Pre-trained weights (gitignored, download separately)
  mt3/                       T5X model, spectrograms, vocabularies
benchmark/                   MAESTRO ground-truth benchmarking
  maestro-v3.0.0.csv         Full dataset metadata (1276 pieces)
  chopin_op10/               Catalog + GT MIDI for Chopin Études Op. 10
  chopin_op25/               Catalog + GT MIDI for Chopin Études Op. 25
  liszt_transcendental/      Catalog + GT MIDI for Liszt Transcendental Études
IMPROVEMENTS.md              Full improvement catalog with priority matrix
STATUS.md                    Complete implementation tracker
```

## Known Limitations

### 1. Quantization

MIDI stores raw timing (e.g., a note held for 0.48 beats). Score backends must
snap this to a musical value (quarter note? dotted eighth?). Heuristics work
well for metronomic playing but degrade with rubato or expressive timing.

### 2. Staff splitting (left/right hand)

Piano scores use treble and bass clef. MIDI has no hand assignment. The
MIDI2ScoreTransformer backend handles this via a learned `hand` stream
(correctly splits into 2 parts). The music21 and musescore backends produce
only 1 part from single-track MIDI.

### 3. Voice separation

Within a single staff, multiple melodic lines must be separated into distinct
voices for correct notation. MIDI2ScoreTransformer predicts a `voice` stream;
music21 uses basic heuristics.

### 4. Key and time signature inference

MIDI files contain no key or time signature metadata. MIDI2ScoreTransformer
predicts these per-note but can be noisy — a majority-vote post-correction
overrides outlier time signatures. On ground-truth MAESTRO MIDI, the model
gets time signature correct on simple pieces but struggles on complex ones
(e.g. Liszt Mazeppa: predicted 3/4 instead of 4/4).

### 5. Enharmonic spelling

MIDI note 66 could be F# or Gb — the correct choice depends on harmonic
context. MIDI2ScoreTransformer predicts accidentals per-note with constraints;
music21 uses key-aware heuristics.

### 6. Pedal interpretation

Sustain pedal extends note durations in MIDI, but scores notate pedal markings
separately. The pipeline does not distinguish pedaled sustain from written
duration, which can produce excessively long tied notes.

### 7. MIDI round-trip quantization

Transcribers like hFT produce float-precision timestamps that get quantized to
MIDI integer ticks when written via pretty_midi. The hFT + transformer pipeline
bypasses this via direct note handoff (`tokenize_notes()`). MT3 still requires
the MIDI file as a handoff between JAX and PyTorch venvs.

## References

- [MIDI2ScoreTransformer](https://github.com/TimFelixBeyer/MIDI2ScoreTransformer) — Beyer, ISMIR 2024
- [hFT-Transformer](https://github.com/sony/hFT-Transformer) — Sony, ISMIR 2023
- [MT3](https://github.com/magenta/mt3) — Hawthorne et al., ISMIR 2021 / ICLR 2022
- [Transkun](https://github.com/Yujia-Yan/Transkun) — Yan & Duan, NeurIPS 2021 / ISMIR 2024
- [Basic Pitch](https://github.com/spotify/basic-pitch) — Spotify
- [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro) — Google Magenta
