# Joint Lab Presentation — Full Script & Slide Guide
### "From Idea to Sheet Music: Two Halves of One Music-AI System"
**Presenters:** Harry (audio/performance → score) + Antoine (text/melody → music)
**Audience:** Prof. Bayen + lab · **Target length:** ~30 min talk + ~10 min demos/Q&A

---

## HOW TO USE THIS DOCUMENT
For every slide you get three things:
- **🖼️ ON SLIDE** — exactly what to put on the slide (title + bullets + visuals).
- **🎙️ SAY (verbatim)** — a word-for-word script you can read or paraphrase.
- **🗣️ WHO** — which of you speaks it.
- **🔬 TECH POCKET** — an *optional* ~10-second technical aside for the engineers in the room (Prof. Bayen). They're self-contained: **skip them** if the room is non-technical or you're tight on time, **drop one in** if you see people leaning forward. Deeper material lives in **Appendix D — Technical backup slides** (pull up only if asked in Q&A).

Three **🔴 DEMO** blocks are scripted with setup, the live steps, and a pre-recorded fallback.

### ⚠️ THINGS TO FINALIZE BEFORE PRESENTING (I couldn't confirm these)
1. **Demo 3 (Lyria + Songscription) specifics** — I wrote it from your description ("text → Lyria audio → Songscription score"). Tell me: is it a script, a notebook, or a web UI? Live or pre-recorded? Which Lyria access path (Vertex AI / Gemini API / MusicFX)? I left `[CONFIRM]` tags where I'm guessing.
2. **Talk length** — script is built for ~30 min; say if you want a 15-min version (I marked **[CUT-FOR-SHORT]** on droppable slides).
3. **Speaking split** — I assigned Antoine his section, you yours, and "BOTH" for framing. Adjust freely.
4. **Antoine's live numbers** — his v2 is mid-training (loss ~0.076, epoch 2); confirm the latest before quoting.

---

# ACT STRUCTURE (high-level flow)

| # | Section | Speaker | ~min | Demo |
|---|---|---|---|---|
| 0 | The Vision (idea → sheet music) | BOTH | 4 | — |
| 1 | Antoine — Generating music (CWT) | Antoine | 7 | 🔴 Demo 1 |
| 2 | Harry — Reading music (audio→score) | Harry | 9 | 🔴 Demo 2 |
| 3 | The Convergence (why we're one lab) | BOTH | 5 | — |
| 4 | The Full Loop + where this goes | Harry | 5 | 🔴 Demo 3 |
| 5 | Close / ask / Q&A | BOTH | — | — |

**The one-sentence thesis (memorize this):**
> *"Music AI has two halves — creating music and notating music — and our lab is building both; this talk shows that they're not two projects but one, because they converge on the same hard problem: a faithful, learnable representation of musical notation."*

---

# ═══════════════════════════════════════════
# ACT 0 — THE VISION  (BOTH, ~4 min)
# ═══════════════════════════════════════════

## Slide 0.1 — Title
**🖼️ ON SLIDE**
- Title: **From Idea to Sheet Music**
- Subtitle: *Two halves of one music-AI system*
- Names: Harry Ilanyan · Antoine [last name] · Bayen Lab
- A single image: a text prompt on the left → a beautiful engraved score on the right, with a faint "?" arrow between them.

**🎙️ SAY (Harry):**
> "Hi everyone. Antoine and I have been working on what look like two separate projects — he generates music, I turn performances into sheet music. Today we want to show you they're actually one project, and that they meet in a surprising place. Let's start with the dream."

---

## Slide 0.2 — The Dream
**🖼️ ON SLIDE**
- Big arrow flow: **Idea (text)** → **Music** → **Sheet Music you can read & play**
- One line: *"Type 'a melancholic Chopin-style nocturne' → get audio AND a clean, playable score."*

**🎙️ SAY (Antoine):**
> "The north star — Prof. Bayen's direction — is this: a musician, a student, anyone, types an idea in plain language and gets back real, performable sheet music. To get there you need two capabilities. One: a model that **creates** music. Two: a model that can take **sound** — a performance, a recording, or generated audio — and write it down as a **correct score**. I build the first half. Harry builds the second. And the punchline of today is that the hard part of both is the *same* part."

---

## Slide 0.3 — The Map (the lab on one slide)
**🖼️ ON SLIDE**
- A diagram with two lanes converging on a shared box:
  - **GENERATION (Antoine):** text/melody → symbolic music
  - **TRANSCRIPTION (Harry):** audio/performance → engraved score
  - Both arrows point into a shared box labeled: **"High-fidelity music NOTATION representation"** (rhythm, tuplets, voices, hands)
- Caption: *"Different inputs, same core problem."*

**🎙️ SAY (Harry):**
> "Here's the map. Antoine's half goes from an idea to music. My half goes from sound to a score. But both of us, independently, ran straight into the same wall: **how do you represent music notation so a neural network can actually learn it** — the rhythm, the tuplets, the two hands, the voices. That shared box is the real story. Hold that thought; we'll come back to it and show you we even built the *same kind of solution* without coordinating."

---

# ═══════════════════════════════════════════
# ACT 1 — ANTOINE: GENERATING MUSIC  (Antoine, ~7 min)
# ═══════════════════════════════════════════

## Slide 1.1 — The Problem (generation)
**🖼️ ON SLIDE**
- Title: **Teaching a model to compose like a pianist**
- Bullets:
  - Goal: given a short musical prompt (a melody, a few bars), **generate an idiomatic piano continuation / harmonization**.
  - Hard because: piano music is **two hands, polyphonic, with ornaments, dynamics, pedaling, rubato** — not a single melody line.
- Visual: a 2-bar Chopin prompt → "?" → a full piano texture.

**🎙️ SAY (Antoine):**
> "My job is generation. Give the model a melody or a few opening bars, and it should continue or harmonize them the way a real pianist would — two hands, chords, ornaments, phrasing. That's much harder than generating a single melody line, because piano music is deeply polyphonic and full of notation detail."

---

## Slide 1.2 — The Representation: Compound-Word Tokenization
**🖼️ ON SLIDE**
- Title: **CWT = Compound-Word Transformer**
- Key idea: *every musical event = one "compound token" made of **14 parallel heads***, each a different musical dimension.
- A table/graphic of the heads (show ~8 of the 14 so it's readable):
  - Type (note/chord/rest/barline/grace) · Duration · Pitch(+octave+accidental) · Ornament · Articulation · Tie · Slur · **Hand (RH/LH)** · Beam · Dynamic · Pedal · Tempo …
- Format: trained on **Humdrum \*\*kern** (compact text notation; the KernScores tradition).

**🎙️ SAY (Antoine):**
> "The core design is the tokenizer. Instead of one giant vocabulary, I represent each musical event as a **compound token** — fourteen parallel 'heads', each describing one dimension: what type of event, its duration, its pitch, ornament, articulation, which **hand** plays it, beaming, dynamics, pedal, tempo. The model predicts all fourteen at once. I work in **kern**, a compact text format for notation with a big ready-made corpus. Remember this 'fourteen parallel heads' idea — it's going to show up again on Harry's side."

**🔬 TECH POCKET** *(optional, ~10s — for the technical crowd)*
> "Technically: each event is **14 categorical heads predicted jointly**, and the whole vocabulary across all heads is only a *few hundred* tokens — versus 20,000-plus for raw kern. That compression is why an entire piece fits in a 2,048-token context window."

---

## Slide 1.3 — The Model
**🖼️ ON SLIDE**
- Title: **The architecture**
- Bullets:
  - ~**16.2M-parameter** Transformer, 10 layers, **RoPE** positional encoding.
  - **3-stage cascaded decoding:** predict *what kind* of event → then duration/pitch/dynamics → then ornaments conditioned on duration & pitch (trills depend on note length; staccato on pitch+duration).
  - Hand-aware: the model declares **RH vs LH** at every step and interleaves them.
- Small "v1 → v2 → v3" timeline chip.

**🎙️ SAY (Antoine):**
> "The model is a ~16-million-parameter Transformer with rotary position encoding. The clever bit is **cascaded decoding in three stages** — it first decides *what kind* of event this is, then the core attributes like pitch and duration, then the fine ornaments *conditioned* on those — because, musically, a trill only makes sense once you know the note's length. And it explicitly tracks which hand is playing. It's gone through three generations: v1 proved the concept, v2 — training right now — adds pedaling, tempo, and richer conditioning, and v3 experiments with compressing whole chords into single tokens."

**🔬 TECH POCKET** *(optional, ~10s)*
> "The three stages model the *conditional dependencies between attributes* explicitly — ornaments are predicted **conditioned on** the already-decided duration and pitch, not as if the 14 heads were independent. Mechanically the embedding concatenates all 14 head-vectors and projects them down: `concat(14 × 320) → 320`."

---

## Slide 1.4 — The Data
**🖼️ ON SLIDE**
- Title: **~219,000 piano scores**
- Sources: PDMX + ASAP + Chopin first editions + **KernScores**.
- One stat: *">99.5% of pieces fit in a 2,048-token window"* (so the model sees whole pieces).
- Honest note: *"v2 is mid-training (epoch 2, loss ~0.076); v1 already produces coherent 6-bar continuations."*

**🎙️ SAY (Antoine):**
> "It trains on about 219,000 piano scores — one of the larger symbolic piano corpora assembled for generation — pulled from public collections and cleaned up. v2 is training as we speak; v1 already generates musically coherent passages from a short prompt. Let me show you."

---

## 🔴 DEMO 1 — ANTOINE GENERATES MUSIC  (~2–3 min)
**🗣️ WHO:** Antoine
**Setup (before the talk):** have a checkpoint loaded; pre-open MuseScore; pick a prompt (Chopin Nocturne Op.9 works best per the repo notes).
**Live steps:**
```bash
cd music-cwt/music_cwt/v2
python gen_and_convert.py \
  --prompt-krn ../prompt/nocturne_op9n1.krn --prompt-measures 6 \
  --temperature 0.8 --top-k 30 --max-tokens 1200 \
  --output generated/demo_continuation.krn
# → opens generated/demo_continuation.mxl in MuseScore
```
**What to show:** the **6-bar Chopin prompt**, then the **AI-generated continuation** rendered as a real score; hit play in MuseScore so they *hear* it.
**🎙️ SAY:** *"These first six bars are the human prompt — Chopin. Everything after the line is the model. Notice it keeps two real hands, repeats and varies the theme, and stays in key."*
**🟡 FALLBACK (if live fails):** pre-render two PDFs/MP3s (prompt + generation) and play the audio. **Always have this ready.**

---

## Slide 1.5 — Where Antoine's half stands
**🖼️ ON SLIDE**
- ✅ Working: compound tokenizer, v1 generation, kern→MusicXML→MuseScore pipeline.
- 🚧 In progress: v2 training (pedal/tempo/3-stage), held-out evaluation set, multi-staff (3+ voices).
- 🎯 Next: style control (per-composer), longer coherence.

**🎙️ SAY (Antoine):**
> "So: generation works end-to-end today, v2 is leveling it up, and the open frontiers are longer-range coherence and finer stylistic control. Now — Harry's half, which is where the surprise lands."

---

# ═══════════════════════════════════════════
# ACT 2 — HARRY: READING MUSIC  (Harry, ~9 min)
# ═══════════════════════════════════════════

## Slide 2.1 — The Problem (transcription)
**🖼️ ON SLIDE**
- Title: **From a performance to a readable score**
- Visual: waveform → piano-roll (MIDI) → engraved sheet music.
- One line: *"'What was played' (messy, expressive timing) → 'what a musician reads' (quantized rhythm, beams, voices, hands)."*

**🎙️ SAY (Harry):**
> "My half is the reverse direction: take *sound* — a piano performance — and write it down as a correct, readable score. The catch is that a performance and a score are very different objects. A performance is expressive, with human rubato and micro-timing. A score is clean and quantized — bar lines, beams, two staves, voices. Going from one to the other is a hard structured-prediction problem: you have to *infer the underlying grid* from messy human timing."

---

## Slide 2.2 — The Pipeline (two blocks)
**🖼️ ON SLIDE**
- Title: **Two blocks**
- Diagram:
  - **Block 1 — Audio → MIDI** (transcription): Basic Pitch / **hFT-Transformer** / MT3 / Transkun
  - **Block 2 — MIDI → Score** (notation): **MIDI2ScoreTransformer** (Beyer & Dai, ISMIR 2024)
  - → MusicXML → MuseScore → PDF
- Callout: *"We integrated 4 transcribers; hFT-Transformer is best (96.7% onset F1)."*

**🎙️ SAY (Harry):**
> "The system is two blocks. Block 1, audio to MIDI — *which notes were played and when*. I integrated four state-of-the-art transcribers; the best is Sony's hFT-Transformer at about 97% note accuracy. Block 2, MIDI to score — *how do you write those notes down correctly* — uses the MIDI2ScoreTransformer from Beyer and Dai's 2024 paper. And here's a name you'll recognize: that paper's lead author co-founded **Songscription**, the 'Shazam for sheet music' startup. So Block 2 is literally the engine a commercial product runs on — which makes it a great, honest yardstick."

**🔬 TECH POCKET** *(optional, ~10s)*
> "One detail the engineers will like: we added a path that hands the transcriber's **float-precision note events straight into the score model**, skipping the MIDI-file round-trip and its re-quantization. Since the whole task is *about* timing, not throwing timing precision away matters."

---

## Slide 2.3 — Result #1: we reproduced the state of the art
**🖼️ ON SLIDE**
- Title: **Reproduced SOTA — to within noise**
- Small table: **MeanER 11.18 (ours) vs 11.30 (paper)**; all sub-metrics within 0.12.
- Caption: *"MUSTER MeanER = the standard score-transcription error rate, lower is better. This validates our whole eval harness."*

**🎙️ SAY (Harry):**
> "First thing I did was reproduce their result, because you can't improve what you can't measure. On the standard benchmark — 59 performances, the MUSTER error metric — we hit **11.18**, versus their **11.30**. That's within noise. It means our evaluation is trustworthy, and every number after this is measured against a baseline I actually reproduced, not a number from a paper."

**🔬 TECH POCKET** *(optional, ~10s — the rigor)*
> "Two rigor notes. MUSTER is a **C++ aligner** that does a global note-to-note match between two MusicXMLs — so it scores *notation*, not just note counts. And a real gotcha: on Apple Silicon the **MPS backend silently produces degenerate outputs**, so the eval has to run on CPU. Finding that is part of why the 11.18 is trustworthy."

---

## Slide 2.4 — Result #2: on clean piano, commercial quality
**🖼️ ON SLIDE**
- Title: **Clean piano: end-to-end MeanER ≈ 2.75 — commercial quality**
- Side-by-side: a Chopin étude audio clip → our rendered score.
- Honest caveat in small text: *"2.75 is the easy, in-distribution end (clean solo études) — **not** a like-for-like win over the product: the score model is the same open checkpoint Songscription runs, and the apples-to-apples number is the 11.18 reproduction. The hard Romantic repertoire is the next slide."*

**🎙️ SAY (Harry):**
> "On clean, in-distribution piano — Chopin études from clean recordings — our **full audio-to-score pipeline** scores about **2.75**. One honest caveat so nobody catches us: that's *not* a like-for-like win over Songscription. 2.75 is on easy, clean pieces, while the ~11 baseline is the full-test average across all fourteen composers — and the score model we run is the *same open checkpoint Songscription is built on*. So the fair claim is: **on clean piano, an open pipeline reaches commercial quality, because the score brain is the open part.** The apples-to-apples number is the 11.18 reproduction. The interesting science is the *hard* end — next slide."

---

## Slide 2.5 — The real ceiling: rhythm, not pitch
**🖼️ ON SLIDE**
- Title: **Where it breaks: tuplet rhythm**
- Two facts:
  - It is **not** the transcriber: feeding *perfect* ground-truth MIDI into Block 2 gives the same error (we proved Block 1 isn't the bottleneck).
  - It is **not** pitch: the notes are right. It's **where the notes sit in time** — triplets and tuplets.
- Killer stat (Liszt "Mazeppa," ~49% tuplets): notes ≈ correct, but **tuplets produced ≈ half of ground truth → wrong meters → 70% too many measures → score alignment collapses.**

**🎙️ SAY (Harry):**
> "When the music gets dense — late-Romantic, virtuoso — the pipeline breaks, and I spent a long time finding out exactly where. Two surprises. One: it's **not the transcriber**. I fed the model *perfect* ground-truth MIDI and it failed the same way — so Block 1 is fine. Two: it's **not pitch** — the notes are right. The failure is **rhythm**: specifically **triplets and tuplets** — notes that sit *between* the obvious beats. On a piece that's half tuplets, the model produces only about half the tuplets it should, which forces every note onto a straight grid, which makes it invent the wrong time signature, which inflates the measure count by 70%, and the whole alignment falls apart. Pitch-perfect, rhythm-broken."

**🔬 TECH POCKET** *(optional, ~10s — the mechanical root, the best one to use)*
> "Here's the mechanism. The rhythm grid is **one twenty-fourth of a quarter note**. Dyadic values land on multiples of three — a quarter is 24, an eighth is 12 — and triplets land *off* them, at 8 and 16. So 'is this a triplet' becomes 'is this onset on a non-multiple-of-three bucket,' and those buckets are individually **rare** in the data. The model learns to never bet on them."

---

## Slide 2.6 — The deep finding (the science)
**🖼️ ON SLIDE**
- Title: **Why is tuplet placement so hard? A careful answer.**
- The diagnostic: the model places **0%** of notes at triplet positions on hard pieces (ground truth: ~4.8%).
- A compact matrix (the thing that makes this a *result*, not a complaint): across every axis, **no lever hit the *correct* rate** — most collapsed to ~0%, one *over*-shot to 19.9% (still wrong):
  - data (reshape → 19.9% over; self-training; paired distillation w/ 24.7% triplet targets → ~0%)
  - representation (beat-relative tokenization → 0%)
  - loss (tuplet-weighting ×5 → 0%)
  - training (warm-start *and* from-scratch; AR *and* non-AR; the from-scratch run was **undertrained**)
- Conclusion (boxed): *"We matched the released model's architecture and validation loss, and came within ~1 MUSTER point — but no lever hit **correct-rate** placement (reshaping over-shot to 19.9%; everything else under-shot). The leading explanation is the **data** — our corpus is ~1.7% tuplets. The decisive test (a tuplet-aware representation trained to convergence on tuplet-rich data) is still open, so this is 'not recovered within our budget,' not 'impossible.'"*

**🎙️ SAY (Harry):**
> "So I went deep — and this is the part I'm proudest of, even though it's a *negative* result, because it's careful. I built a diagnostic: what fraction of notes does the model place at triplet positions? Ground truth is about 5%. Then I tried to fix it from every angle — reshaping the data, self-training, a brand-new tuplet-aware representation, cranking the loss, even rebuilding the data engine and training from scratch on 23,000 paired pieces where a quarter of the targets were explicitly triplets. The lever that *over*-shot hit 20% — wrong rate, worse score. Everything else under-shot or collapsed to zero. **Nothing hit the *correct* rate.** We matched the released model's validation loss and got within about a point of its benchmark — but never its correct-rate placement. The leading explanation is the **data**: our corpus is only about 2% tuplets. So this isn't a dead end — it's a *data problem*, which tells us exactly what to fix, and it's the strongest argument for just *using* the released model as our engine."

**🔬 TECH POCKET** *(optional, ~10s — the representation fix)*
> "The fix I built — call it B2 — re-expresses an onset as **within-quarter position plus an integer quarter-index**, so a triplet becomes the *same* common bucket in every beat. I proved that recoding is **lossless** and concentrates **100% of triplets into shared buckets**. And the model *still* wouldn't learn them without the data — which is precisely how we know the representation is necessary but not sufficient."

> **[CUT-FOR-SHORT]** *you can compress 2.5 + 2.6 into one slide if time is tight; keep the "0% placement, tried everything" punchline.*

---

## 🔴 DEMO 2 — HARRY TRANSCRIBES A PERFORMANCE  (~2–3 min)
**🗣️ WHO:** Harry
**Setup:** have a clean Chopin clip ready; pre-render the output so you can show it instantly even if you also run live.
**Live steps:**
```bash
cd musicML
python transcribe.py benchmark/chopin_op10/audio/Op10_No4_CsharpMinor.wav \
  -t hft -b transformer --keep-midi
# → audio → hFT MIDI → MIDI2ScoreTransformer → MusicXML → PDF
```
**What to show:** play ~10 s of the **audio**, then reveal the **rendered score PDF** (in `outputs/transformer/`), and point out the two clean staves, correct key, beams.
**🎙️ SAY:** *"That was the raw recording. With one command it becomes this — two-hand notation, correct key signature, beamed and readable. On clean piano this is essentially product-quality."*
**🟡 FALLBACK:** show the pre-rendered `benchmark/chopin_op10/transformer/` PDF (already in the repo). **Have it open in a tab.**

---

# ═══════════════════════════════════════════
# ACT 3 — THE CONVERGENCE  (BOTH, ~5 min)  ← the heart of the talk
# ═══════════════════════════════════════════

## Slide 3.1 — We built the same thing without coordinating
**🖼️ ON SLIDE**
- Title: **Same idea, both halves: multi-head compound tokens**
- Side-by-side:
  - **Antoine (generation):** 14 parallel heads — type, duration, pitch, ornament, hand, dynamics, pedal…
  - **Harry (transcription):** 13 parallel streams — offset, duration, downbeat, pitch, accidental, voice, stem, **hand**…
- Caption (big): *"Two people, two directions, independently chose to represent music as a set of parallel attribute streams. That's not a coincidence — it's the right abstraction for notation."*

**🎙️ SAY (Harry):**
> "Here's the surprise we promised. Antoine and I never coordinated our representations. He encodes each musical event as **14 parallel heads**. The transcription model I work on encodes each note as **13 parallel streams** — pitch, duration, voice, stem, hand, and so on. Same idea. Two people, opposite directions, both landed on **multi-head compound tokenization** because it's the natural way to represent notation: a note isn't one symbol, it's a *bundle of attributes*."

**🎙️ SAY (Antoine):**
> "And we both hit the *same wall*: rhythm and notation fidelity. My generation has to get tuplets, voices, and hands right; Harry's transcription dies on exactly those. The hard problem isn't generation versus transcription — it's the **representation of notation** sitting underneath both."

**🔬 TECH POCKET** *(optional, ~10s — the deepest convergence; great for an engineering audience)*
> "And it goes one layer deeper than the tokens. **Both models are rotary-position transformers** — Antoine's CWT and the MIDI2ScoreTransformer are *both* RoFormers, the same positional encoding. So it's compound multi-head tokens **and** RoPE, chosen independently on both sides — two people rediscovering the same recipe."

---

## Slide 3.2 — So it's one system, not two
**🖼️ ON SLIDE**
- Title: **One system: the data engine + the generator**
- Diagram (the unifying picture):
  - **Audio (unlimited)** → [Harry: transcription] → **score corpus at scale** → [Antoine: generative pretraining] → **text → score**
- Two captions:
  - *"Scores are scarce; audio is unlimited. Transcription manufactures training data for generation."*
  - *"Both stages share one notation representation — owning it is the lab's real moat."*

**🎙️ SAY (Harry):**
> "Once you see the shared representation, the two projects click into one system. The bottleneck for *any* text-to-score generator is **paired training data**, and scores are scarce. But audio is effectively unlimited. So my transcription pipeline is a **data engine**: it manufactures large score corpora from audio to pretrain Antoine's generator. That's exactly the moat the commercial players have — synthetic data at scale. My tuplet finding even tells us *which* data to trust and which dense-rhythm cases to down-weight so we don't teach Antoine's model bad rhythm."

---

## Slide 3.3 — What each of us owns
**🖼️ ON SLIDE**
- Title: **Division of labor that actually composes**
- Antoine: the **generator** (text/melody → music) + the symbolic-music modeling.
- Harry: the **data engine** (audio → score at scale) + **the notation-representation/fidelity** that both models depend on.
- One line: *"Harry's role isn't 'the transcription guy' — it's the owner of the shared notation representation."*

**🎙️ SAY (Antoine):**
> "Concretely: I own the generative model and the music modeling. Harry owns the data engine *and* — because of the tuplet work — the notation-fidelity layer that both of our models stand on. That's a division of labor that genuinely composes into the lab's end goal instead of two parallel tracks."

---

# ═══════════════════════════════════════════
# ACT 4 — THE FULL LOOP + WHERE THIS GOES  (Harry, ~5 min)
# ═══════════════════════════════════════════

## Slide 4.1 — The vision, end to end (today, off the shelf)
**🖼️ ON SLIDE**
- Title: **Text → Music → Score — the whole loop**
- Flow: **Text prompt** → **Google Lyria** (text → music *audio*) → **transcription** (audio → score) → **editable sheet music**
- Caption: *"A concept demo of the end goal, stitched from today's best pieces — Lyria for generation, our transcription for notation."*

**🎙️ SAY (Harry):**
> "To make the end goal tangible, I built a concept demo of the *whole* loop using today's best off-the-shelf pieces. You type a description. **Google Lyria** — DeepMind's text-to-music model — turns that text into actual *audio*. Then the transcription side turns that audio into a *score* you can read and edit. That's the dream from slide one, working end to end, today."

---

## 🔴 DEMO 3 — THE FULL LOOP (TEXT → AUDIO → SCORE)  (~3 min)
**🗣️ WHO:** Harry
> **`[CONFIRM]` Fill in your real implementation — below is the scripted flow from your description.**
**Scripted steps:**
1. Type a prompt, e.g. *"a gentle, romantic solo-piano nocturne in A minor."*
2. **Lyria** generates ~30 s of piano **audio** `[CONFIRM access path: Vertex AI Media Studio / Gemini API / MusicFX]`.
3. Feed that audio into the transcription step (**Songscription** in the concept version, and/or **my own pipeline** as the "real" version) → **sheet music**.
4. Show the score; if possible, play the Lyria audio next to the rendered score.
**🎙️ SAY:** *"Text in… Lyria gives us audio… and the transcriber writes it down. This is the concept; the part I actually engineered and measured is that middle-to-right transcription step — which reproduces, and on clean piano matches, the engine Songscription is built on."*
**🟡 FALLBACK:** pre-record the screen capture end-to-end (Lyria can be slow/rate-limited live — **strongly recommend pre-recording this one**).
**Framing line to land:** *"Demo 1 was Antoine's real generator. Demo 2 was my real transcriber. Demo 3 is the vision — and it shows exactly where our two real systems slot in to replace the off-the-shelf parts."*

---

## Slide 4.2 — Where this goes (Bayen's direction)
**🖼️ ON SLIDE**
- Title: **Roadmap**
- Three bullets:
  1. **Data engine:** transcription manufactures score corpora to pretrain the generator (use the released model as the placement-capable engine; filter dense-tuplet cases).
  2. **Shared notation representation:** converge our 13/14-head schemes; evaluate a **rhythm-honest, LLM-friendly format (e.g., ABC)** that both halves use — and that makes *text* conditioning natural.
  3. **Close the loop:** replace Lyria/Songscription in the demo with Antoine's generator + my pipeline → a true **text → score** system the lab owns.

**🎙️ SAY (Harry):**
> "Three concrete next steps. One: stand up the data engine — use transcription to mass-produce training scores for Antoine's generator, with my tuplet findings telling us what to trust. Two: converge our two representations into one shared, rhythm-honest format — which is also the thing that makes *text* conditioning natural, the whole point of the lab. Three: close the loop — swap Lyria and Songscription in that last demo for Antoine's generator and my transcriber, so the text-to-score system is fully ours."

---

## Slide 4.3 — Summary / the ask
**🖼️ ON SLIDE**
- Title: **Takeaways**
- Bullets:
  - Two halves — **generation (Antoine)** and **transcription (Harry)** — both working end-to-end with live demos.
  - They're **one system**: same representation idea (compound multi-head tokens), same hard problem (notation fidelity), composing into **text → score**.
  - Real results: reproduced SOTA (**11.18**, within noise of the paper), matched commercial quality on clean piano (**2.75** end-to-end), and a **rigorous characterization** of the tuplet ceiling (it's a *data* problem, not a dead end).
  - The ask: **[CONFIRM with Harry — e.g., compute for the data engine / a decision on the shared representation / a green-light to merge tracks]**.

**🎙️ SAY (BOTH):**
> (Harry) "To wrap: you saw two real systems — Antoine's generator and my transcriber — and a concept demo of the full text-to-score loop."
> (Antoine) "And the real message is that they're one system: we independently built the same kind of representation and hit the same hard problem, so they belong together."
> (Harry) "What we'd love from you: **[the ask]**. Happy to take questions — and to play more demos."

---

# ═══════════════════════════════════════════
# APPENDIX A — LIKELY QUESTIONS FROM PROF. BAYEN (prep these)
# ═══════════════════════════════════════════

**Q: "Your 2.75 vs 11 — isn't that cherry-picked?"**
> "Yes, deliberately, and I flag it: 2.75 is the clean in-distribution end; the 11 is the full-test average. The honest, apples-to-apples number is the 11.18 reproduction. The 2.75 shows the *ceiling* of what's possible on the music it's designed for; the tuplet finding shows the *floor* on hard music."

**Q: "Why couldn't you reproduce the tuplet placement? Did you just not train long enough?"**
> "Honest answer: partly — our *capstone* from-scratch run was undertrained, so I can't claim it's impossible. What I can say is sharper: a model that matched the released *validation loss* still showed zero correct-rate placement, and pushing it with data, representation, and loss never hit the correct rate either. The leading explanation is the **data** — our corpus is ~2% tuplets, far poorer than theirs. So it's 'not recovered within our compute and data budget, and the data is the binding constraint' — which points to a concrete fix: a tuplet-rich corpus trained to convergence. The one cheap thing I haven't done is email the author for the corpus/schedule."

**Q: "Why kern / why two different tokenizers? Shouldn't you unify now?"**
> "Exactly the plan — slide 4.2. We independently validated that multi-head compound tokenization is right; the next step is converging on one shared scheme, and evaluating a text-native format like ABC so the *generation* side gets natural language conditioning for free."

**Q: "Is the data engine real or aspirational?"**
> "The pieces are real: a working transcription pipeline that reproduces SOTA, plus a render/pairing pipeline that already built a 23,000-piece paired corpus in minutes. What's aspirational is scaling it and wiring its output into Antoine's pretraining — that's the ask."

**Q: "What's genuinely novel vs. using off-the-shelf models?"**
> "Three things: (1) the rigorous, controlled characterization of the tuplet-placement ceiling — nobody has published that; (2) the data-engine framing that turns unlimited audio into scarce score data; (3) the convergence insight — that generation and transcription share one representation, which reframes the lab's whole architecture."

**Q (to Antoine): "How do you evaluate generation quality?"**
> "Right now: loss curves + token-distribution metrics + listening tests on standard prompts. A held-out, non-overlapping test set is in progress (it's on the todo). Long-range coherence is the honest open weakness."

---

# ═══════════════════════════════════════════
# APPENDIX B — DEMO LOGISTICS CHECKLIST
# ═══════════════════════════════════════════
- [ ] **Pre-record all three demos** as video, even if you also run live. (Especially Demo 3 — Lyria can rate-limit.)
- [ ] Demo 1: checkpoint downloaded; MuseScore open; prompt chosen (Chopin Op.9 nocturne); audio export of one generation ready.
- [ ] Demo 2: `venv311` active; hFT checkpoint present; one clean Chopin clip; pre-rendered PDF open in a tab as fallback.
- [ ] Demo 3: `[CONFIRM]` Lyria access working; a known-good text prompt that produces clean piano; pre-recorded screen capture as primary or fallback.
- [ ] A single "demos" folder with: prompt files, generated outputs, rendered PDFs, audio clips — so nothing depends on a live network.
- [ ] Speaker handoff cues marked on each slide (WHO).

---

# ═══════════════════════════════════════════
# APPENDIX C — FACT-CHECK NOTES (so you don't get caught)
# ═══════════════════════════════════════════
- **MeanER 11.18 vs 11.30** — our reproduction vs the Beyer & Dai (ISMIR 2024) paper; verified, within noise. The "MUSTER" metric is the standard score-transcription error rate.
- **The Songscription comparison (now framed correctly in the slides):** 11.30 is the *paper's* number for the model **Songscription is built on** (co-founder Tim Beyer = the paper's lead author) — *not* a number Songscription publishes. Our **2.75** is on clean piano (easy subset), end-to-end, running that same open checkpoint — so the claim is **"matches commercial quality on clean piano," not "beats Songscription."** The apples-to-apples number is the 11.18 reproduction. If pushed, concede this proactively — it's the strongest move.
- **CWT = Compound-Word Transformer** (Antoine's model), **not** Continuous Wavelet Transform. (If anyone asks: wavelets are an *audio* analysis tool; this is symbolic generation.)
- **Lyria** = Google DeepMind **text → music *audio*** (not notation). Versions move fast — say "Lyria 2 (instrumental, via Vertex AI)" unless you're using the newer Lyria 3 / RealTime; confirm your access path before the talk.
- **Songscription** = audio/recording (or YouTube) → editable sheet music/MIDI/MusicXML; freemium; based on the Beyer & Dai architecture.
- **hFT-Transformer** = Sony, ISMIR 2023, ~96.7% onset-F1; best of the four transcribers we integrated.
- **Antoine's v2** is mid-training — **confirm the latest epoch/loss the morning of the talk** before quoting.

---

# ═══════════════════════════════════════════
# APPENDIX D — TECHNICAL BACKUP SLIDES (pull up only if asked in Q&A)
# ═══════════════════════════════════════════
*Not part of the main flow. These are the "go deeper" slides for an engineering audience — have them ready as hidden slides.*

## D1 — The two tokenizers, side by side (the convergence, in full)
- **Antoine — CWT, 14 heads:** Type · Duration · Pitch(+octave+accidental) · Ornament · Articulation · Tie · Slur · Phrase · Voicing(stem) · Beam · **Hand (RH/LH)** · Dynamic · Pedal · Tempo. Vocab ≈ a few hundred tokens; embedding `concat(14 × 320) → 320`; **RoPE**; 3-stage cascaded decode; ~16.2M params; autoregressive.
- **Harry — MIDI2ScoreTransformer, 13 streams + pad:** offset · downbeat · duration · pitch · accidental · keysignature · velocity · grace · trill · staccato · voice · stem · **hand** · + a `pad` gate stream. **RoFormer** enc-4 / dec-4, hidden 512, 8 heads, **~32.6M params**; non-autoregressive (bidirectional mask-predict).
- **Shared, unplanned:** multi-head **compound tokens** + **RoPE**, on both sides. Both quantize rhythm on a grid and both must solve hand/voice assignment.

## D2 — MUSTER, in detail
- Python wrapper (`muster/muster.py`) around a **C++ aligner** over two MusicXMLs. Six error rates: PitchER, MissRate, ExtraRate, OnsetER, OffsetER → **MeanER** (their mean). Plus engraving sub-metrics (note spelling, voice assignment, stem direction, staff assignment) and voice P/R/F1. **Aborts on 3-staff scores** (limits OOD eval to 2-staff). Our 11.18 matched the paper on *all five* sub-metrics within 0.12.

## D3 — The 1/24 grid + the placement diagnostic
- Grid = **1/24 of a quarter note**. Dyadic = multiples of 3 (quarter 24, eighth 12, 16th 6); triplet/sextuplet = off-multiples (8, 16, 4, 20). `diag_offset_phase.py` measures the fraction of onsets at triplet phases: **GT ≈ 4.78%, our models 0%**.
- **B2** = within-quarter offset + integer `quarter_idx`. Verified: **lossless** round-trip, **100%** triplet concentration into shared buckets, non-B2 byte-identical. Conclusion: representation **necessary, not sufficient**.

## D4 — The exhaustive placement matrix + the data engine
- Inference cleanup → 11.79→**11.41** (only robust gain); global logit-shift → over-emits; corpus reshape → **19.9%** (over-shoots, regresses); self-training → 0.9%; tuplet-weight ×5 → 0%; B2 → 0%; paired distillation (2.87% targets) → 0%; rendered paired (**24.73%** targets) → 0% (undertrained); from-scratch AR (val 1.3, undertrained) + non-AR (matched val 0.51, within ~1 MUSTER pt) → 0%.
- **Data engine:** 23,780 PDMX scores rendered → valid per-beat-aligned paired data in ~11 min; tuplet-rich subset carries 24.73% placement.

## D5 — Eval-harness rigor (why the numbers are real)
- **MPS→CPU:** Apple-MPS produced degenerate pad logits → eval forced to CPU. **val ≠ MUSTER:** we re-rank checkpoints by *true* MUSTER (`rerank_by_muster.sh`), not loss. **Leakage audit:** 12/17 eval pieces found in PDMX via *content fingerprints* (transposition-invariant interval n-grams, not titles) → content-dedup tool + blocklist for any future PDMX training. **Pad-threshold bug:** the released `generate()` hard-argmaxed `pad` to binary, making any threshold a no-op — we fixed it (continuous keep-probability + un-zeroed raw streams).
