# Piano Audio → Sheet Music Pipeline

Transcribes piano audio recordings into rendered sheet music using
[Spotify Basic Pitch](https://github.com/spotify/basic-pitch) for audio-to-MIDI
conversion and one of three backends for MIDI-to-score conversion:

- **music21** — rule-based heuristics, rendered via [LilyPond](https://lilypond.org)
- **musescore** — [MuseScore](https://musescore.org) CLI MIDI import + rendering
- **transformer** — [MIDI2ScoreTransformer](https://github.com/TimFelixBeyer/MIDI2ScoreTransformer)
  neural model (ISMIR 2024), rendered via MuseScore

## Setup

### 1. Python environment

Requires Python 3.8–3.11 (basic-pitch constraint). Python 3.11 is required for
the transformer backend.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The transformer backend has additional dependencies (PyTorch, transformers, etc.)
— see `MIDI2ScoreTransformer/requirements.txt`.

### 2. System dependencies

**LilyPond** (required for the music21 backend):
```bash
brew install lilypond          # macOS
sudo apt-get install lilypond  # Linux
```

**MuseScore** (required for the musescore and transformer backends):
```bash
brew install --cask musescore  # macOS
```

The script auto-detects both on your PATH with Homebrew fallback locations.
If LilyPond is not found, the music21 backend falls back to MusicXML export.

## Usage

```bash
python transcribe.py samples/TwinkleTwinkle.mp3                        # music21 backend (default)
python transcribe.py samples/TwinkleTwinkle.mp3 -b musescore           # musescore backend
python transcribe.py samples/TwinkleTwinkle.mp3 -b transformer         # transformer backend
python transcribe.py samples/TwinkleTwinkle.mp3 -o custom/path.pdf     # custom output path
python transcribe.py samples/TwinkleTwinkle.mp3 -m piano.mid           # keep intermediate MIDI at specific path
```

Output defaults to `outputs/<backend>/<input_stem>.pdf` when `-o` is not given.

### Arguments

| Argument | Description |
|---|---|
| `input` | Path to audio file (.wav, .mp3, .ogg, .flac, .m4a) |
| `-o`, `--output` | Output file path. Extension sets format (.pdf or .png). Default: `outputs/<backend>/<input_stem>.pdf` |
| `-m`, `--keep-midi` | Path for intermediate MIDI file. Default: `<input_stem>.mid` beside output |
| `-b`, `--backend` | Score conversion backend: `music21` (default), `musescore`, or `transformer` |

### Backends

| | music21 | musescore | transformer |
|---|---|---|---|
| Approach | Rule-based heuristics | MuseScore MIDI import | Neural seq2seq model |
| Key detection | Krumhansl-Schmuckler | MuseScore internal | Per-note prediction |
| Renderer | LilyPond | MuseScore | MuseScore |
| System deps | LilyPond | MuseScore | MuseScore + PyTorch |
| Device | N/A | N/A | CPU (MPS not supported) |

## Project Structure

```
samples/                     Input audio files
midi/                        Intermediate MIDI files (from Basic Pitch)
outputs/
  music21/                   Scores from the music21 backend
  musescore/                 Scores from the musescore backend
  transformer/               Scores from the transformer backend
MIDI2ScoreTransformer/       Cloned model repo (ISMIR 2024 paper)
  checkpoints/               Pre-trained model weights
  midi2scoretransformer/     Tokenizer, model, inference utilities
transcribe.py                CLI entry point — full pipeline
```

## Test Pieces

Place audio files in `samples/` and run the pipeline. Suggested test pieces, in
order of difficulty:

1. **Twinkle Twinkle Little Star** — Simple melody, mostly monophonic. Search for
   "Twinkle Twinkle Little Star piano solo" for clean recordings.
2. **Bach Prelude in C Major (BWV 846)** — Steady arpeggiated patterns, no wide
   dynamic range. Good test of chord/arpeggio handling.
3. **Chopin Nocturne Op. 9 No. 2** — Expressive, rubato, complex ornamentation.
   Expect rough results; useful for documenting pipeline limitations.

### What to expect

| Piece | Audio→MIDI | MIDI→Score | Overall |
|---|---|---|---|
| Twinkle | Good — clean detection | Good — simple rhythm/key | Readable output |
| Bach Prelude | Good — clear notes | Fair — arpeggios may quantize oddly | Mostly readable |
| Chopin Nocturne | Fair — rubato causes timing drift | Poor — complex rhythm, ornaments | Rough but informative |

## Known Limitations

These are inherent challenges in audio-to-score conversion. The heuristic
backends (music21, musescore) expose all of them; the transformer backend learns
to handle some from training data but is not immune. Documented here for future
work.

### 1. Quantization

MIDI stores raw timing (e.g., a note held for 0.48 beats). Music21 must snap
this to a musical value (quarter note? dotted eighth?). Heuristics work well for
metronomic playing but degrade with rubato or expressive timing.

### 2. Staff splitting (left/right hand)

Piano scores use treble and bass clef. MIDI has no hand assignment. A naive split
at middle C works for simple pieces but fails when hands cross or share a
register. Music21 uses the MIDI track/channel structure when available, but
Basic Pitch outputs a single track.

### 3. Voice separation

Within a single staff, multiple melodic lines (e.g., melody and accompaniment in
the right hand) must be separated into distinct voices for correct notation. This
is an unsolved problem in general; music21 provides basic heuristics.

### 4. Key and time signature inference

MIDI files contain no key or time signature metadata. Music21's `analyze('key')`
uses the Krumhansl-Schmuckler algorithm, which works well for tonal music but can
be fooled by chromatic passages. Time signature detection relies on strong beat
patterns.

### 5. Enharmonic spelling

MIDI note 66 could be F# or Gb — the correct choice depends on harmonic context.
Music21 uses key-aware heuristics, but edge cases (modulations, chromatic
passages) may produce wrong spellings.

### 6. Pedal interpretation

Sustain pedal in MIDI extends note durations, but scores notate pedal markings
separately (Ped. / *). The current pipeline does not distinguish pedaled
sustain from written note duration, which can produce excessively long tied notes.

## Resources for Generation Phase

**[Pianoroll RNN-NADE](https://github.com/magenta/magenta/tree/main/magenta/models/pianoroll_rnn_nade)** —
Not applicable to this pipeline. This is a generative model — it creates new
polyphonic piano music using LSTM + NADE (Neural Autoregressive Distribution
Estimator). You give it a chord or a short primer and it generates a
continuation. It doesn't take audio input, and it doesn't produce notation. It's
for composing, not transcribing.

## Miscellaneous

**[Score2Perf / Music Transformer](https://github.com/magenta/magenta/tree/main/magenta/models/score2perf)** —
Not applicable to this pipeline, but related work. Score2Perf goes Score →
Performance — literally the opposite direction of what we need. It takes a
notated score and generates an expressive MIDI performance with realistic timing
and velocity. The Music Transformer model it's built on is trained on MAESTRO
and generates unconditional piano performances from scratch. Neither mode helps
with audio → score. That said, the Music Transformer architecture and MAESTRO
training methodology are important related work in the broader music-ML space.
