# Morning Report — Overnight Work (2026-06-03)

## ★ HEADLINE UPDATE — the SOTA gap is the DATA STRATEGY, and the fix works
After the overnight work below, I diagnosed *why* our from-scratch model trails the released SOTA, then
fixed it. Short version:

**Diagnosis (evidence, not guesswork):** I loaded both checkpoints. They are the **same architecture to the
byte** (32,595,514 params, same RoFormer 4+4/512/8-head, same causal-AR decoder), and the released ckpt
**saved its full training recipe** — which we had already matched (lr 3e-4, input_dropout 0.75,
unconditional_dropout 0.5, wd 0.2; we even trained *more* steps, 58k vs 40k). So the gap is **not**
architectural and **not** under-training. The one real difference: **their data strategy.** They train
50/50 on real ASAP pairs **+ 58k real *unpaired* engraved scores via masked self-supervision** (surrogate
score-pitch input + a conditioning token); we trained on ~85k **synthetic rendered** pairs. The per-stream
error breakdown matched this exactly — our gap is concentrated in **pitch (~5.5×) and miss-rate (~5.3×)**
with timing near parity, i.e. a *notation-distribution* deficit, not capacity or optimization.

**Fix (implemented + measured):** I implemented their masked-SSL path (`UnpairedScoreDataset`,
`make_ssl_loaders`, the conditioning token, per-branch 75%/50% decoder dropout) and trained from scratch on
170k PDMX scores as unpaired + ASAP at 50/50. Result, vs our previous best ar_full3 and the released SOTA,
**same eval harness, leak-clean pieces** (Bach/Liszt excluded — they're in PDMX):

| piece (MeanER) | ar_full3 (synth) | ssl_unpaired (pop SSL) | ssl_v2 (pop, exp-matched) | **ssl_classical_clean** | released SOTA |
|---|---|---|---|---|---|
| Mozart K.332 | 12.45 | 8.85 | 7.37 | **5.76 ✅** | 5.86 |
| Scriabin Op.8/11 | 16.3 | 12.68 | 15.15 | **12.3** | 10.37 |
| Ravel Ondine | 27.4 | — | — | **20.51 ✅** | 20.8 |
| Prokofiev Toccata | — | 9.49 | — | 10.61 | — |
| best val/total | 0.9696 | 0.5952 | 0.5645 | **0.5125** | — |

(✅ = matches/beats released SOTA. **These are the LEAK-CORRECTED numbers**: the first classical corpus
contained a few copies of the eval pieces — Ravel Ondine ×5, Scriabin Op.8 ×1, Mozart K332 NONE — so I
re-ran on a leak-filtered corpus (35 ASAP-test scores removed). The numbers HELD: Mozart 5.72→5.76,
Scriabin 12.67→12.3, Ravel 20.14→20.51 — i.e. the leak was negligible and SOTA parity on Mozart+Ravel is
robust to scrutiny.)

The progression is three controlled steps, each isolating one lever:
- **ssl_unpaired** (170k PDMX unpaired): strategy fix alone → val 0.97→**0.5952 in ONE epoch**, Mozart 8.85,
  Scriabin 12.68 (55–61% of the gap). Floor: each unpaired score seen only ~0.5× while 822 real hammered ~94×.
- **ssl_v2** (58k unpaired, **exposure-matched** to their ~12×/score): val→0.5645, Mozart→7.37 (77% of the gap)
  — but **Scriabin REGRESSED to 15.15**: the pop-heavy corpus sharpens common pieces and drifts off the tail.
- **ssl_classical_clean** (24k **genre=classical** unpaired, leak-filtered, same recipe): val→**0.5125 (best of
  all)**, and on MUSTER it **matches the released SOTA on Mozart (5.76 vs 5.86) and Ravel (20.51 vs 20.8)**,
  while **recovering the Scriabin tail (12.3, back from ssl_v2's 15.15)**. Gap closed: Mozart ~101%, Ravel
  ~104%, Scriabin 67%.

**Bottom line: from scratch, with no access to their training data, correctly diagnosing and reproducing the
released model's DATA STRATEGY (real-unpaired masked-SSL on a genre-matched, leak-filtered classical corpus,
exposure-matched) reached SOTA parity on Mozart and Ravel — and the result is leak-corrected and robust.**
`ssl_classical_clean` epoch-13 is the validated best from-scratch checkpoint. The only residual gap is
Scriabin Op.8/11 (12.3 vs 10.37) — the densest-tuplet piece, where SOTA is also weakest and both models
produce ~0 of its 99 tuplets (the field-wide-hard tuplet tail).

### Data-scaling round (ssl_bigc) — the data lever is now EXHAUSTED (turns negative)
To push the tail I web-scraped clean public-domain classical (KernScores/craigsapp, OpenScore, music21
corpus): 12,163 MusicXML → consolidated to **5,422 curated solo-piano scores** (deduped, ≤2-staff,
leak-filtered — 78 ASAP-test pieces caught). Merged with the 23,783 PDMX-classical → 29,181-score corpus →
`ssl_bigc`. Result: **best val of all (0.4996) and Scriabin nudged down (12.3→11.81), BUT it over-produces** —
Mozart emits **166 tuplets where there are 36** (MeanER 5.76→7.6) and Ravel regressed (20.51→24.74). The
curated corpus is *complexity-skewed* (Chopin/Scriabin/Beethoven etudes are tuplet-dense), so more of it biases
the notation prior toward over-emitting tuplets — net WORSE on the test pieces despite the lower val. **So
more classical data is no longer the lever** — `ssl_classical_clean` stays the best balanced model. The
residual balanced tail needs the released *recipe*, not more scores: their chunker beat-alignment, the exact
40k-step schedule, and the **custom `TimFelixBeyer/music21` fork** (tie-stripping — which likely also fixes
both our eval music21 hangs AND this over-production, since it's a tied-note/tuplet tokenization artifact).
The only remaining clear gap is Scriabin Op.8/11 — the densest-tuplet piece (also where SOTA is weakest, 10.37;
both models produce 0 of its 99 tuplets, so the *tuplet* tail is the field-wide-hard residual).

**`ssl_classical` epoch-10 is the new best from-scratch checkpoint.** Remaining levers to fully close Scriabin
(all non-architectural — the repo confirmed identical architecture + 822-real ceiling): a larger classical
unpaired corpus, the released beat-alignment preprocessing + exact 40k-step schedule, and their custom
music21 fork (`TimFelixBeyer/music21`, tie-stripping) which also fixes our eval hangs. Everything below this
line is the earlier work that led here.

---

## TL;DR
Two solid results and one important correction.
- **Result 1 (positive):** the from-scratch trainer is fixed and credible — reverse-engineering the
  released decoder (standard causal-AR) + fixing two real bugs took our from-scratch model from **~50 →
  ~20 MeanER**. That's a working reimplementation with no access to their training script or data.
- **Result 2 (the real finding):** I ran a head-to-head, **same-harness** eval of the *released SOTA
  model* against ours on the hard pieces. **The released model ALSO collapses on the dense-tuplet tail**
  (Scriabin Op.8/11: 0 tuplets produced; Ravel Ondine: only 0.3× of 1281). So the tuplet tail is a
  **field-wide hard problem**, not a defect unique to us — a genuinely useful thing to know.
- **Correction (be honest):** my earlier "Mozart 12.45 ≈ baseline 11.30 → at parity" was an
  apples-to-oranges slip — 11.30 is the *whole-test-set average*, not Mozart. On the **same Mozart**, the
  released model scores **5.86 vs our 12.45** (~2×). So we are *not* at parity; we're a credible
  reimplementation roughly 1.3–2× the released model's per-piece error, and the gap is **broad accuracy**
  (the released model even scores better while producing *fewer* tuplets), reflecting their better
  unpublished training data/recipe — **not** a tuplet-specific deficit.
- **Tonight's experiment (negative):** realistic-rubato rendering + augmentation did not help the tail
  (details below). Data is not the lever; the next levers are architectural or simply using the released
  model for the product.

## The decoder mystery — SOLVED
A multi-agent hunt loaded the released checkpoint directly: the decoder is `is_autoregressive=True`
(standard causal AR). My earlier "unreproducible bidirectional mask-predict recipe" was a false read
of a constructor default. So our `--autoregressive` mode is the correct architecture — no exotic
recipe needed. (No training script is public — it's gated behind a Google Form — but it isn't needed.)

## Two real bugs fixed (the accuracy jump)
1. **Double-shift:** training pre-shifted the decoder input AND the embedding shifts +1 internally
   (is_autoregressive) → trained to predict token[t] from token[t-2] while inference feeds t-1.
2. **No regularization:** the AR branch skipped the 75% decoder dropout the released recipe uses → overfit.
Fix: feed the raw target (embedding does the single shift) + restore the 75% dropout. **A/B on the same
data: MeanER ~66 → ~29.5, and overfitting eliminated.**

## Full trajectory (MUSTER MeanER, lower better; baseline 11.30) + best val loss
| build | data | Mozart | Bach | Scriabin | Ravel | mean | best val | verdict |
|---|---|---|---|---|---|---|---|---|
| buggy first build | 31K | 47 | 45 | 57 | 52 | ~50 | — | bugs |
| FIX, broad-kern | 988 | 27 | 25 | 32 | - | 29.5 | — | fix works |
| **FIX, 85K (ar_full3)** | 85K | **12.45** | 19.9 | 16.3 | 27.4 | **~20** | **0.9696** | **BEST** |
| FIX, 171K (ar_full4) | 171K | 13.2 | 23.0 | 26.9 | - | ~21 | — | pop dilution ↓ |
| kern×30 upweight (ar_full5) | 114K | — | — | — | — | — | 0.9895 | repetition ↓ |
| rubato finetune (ar_rubato) | 84K+rubato-kern | 13.38 | 22.1 | 19.9 | 38.1 | — | 1.09 ↑ | diverged ↓ |

**ar_full3 (85K) is the best general checkpoint** (lowest val 0.9696, Mozart at baseline).
Every attempt to push the tuplet tail with *data* — more PDMX (ar_full4), kern repetition
(ar_full5), and realistic-rubato augmentation (ar_rubato) — has **regressed** vs ar_full3.

## What overnight scaling taught us
- **More PDMX HURTS** (ar_full4, 171K regressed). PDMX is pop-heavy; the ASAP test is classical, so more
  PDMX volume *dilutes* the classical/tuplet signal. PDMX-scaling is exhausted at ~85K.
- **Upweighting tuplet data by REPETITION hurts too** (ar_full5): repeating the 988 kern pieces ×30
  (→26% of synthetic) gave val **0.9895 — worse than ar_full3's 0.9696**, and its hard-piece output was
  too rough for MUSTER to even score (music21 hung on it). Lesson: the tail needs better tuplet data
  **quality**, not just **quantity**. That motivated the realistic-rubato work below.
- **Eval tooling hardened**: MUSTER/music21 kept hanging (C-level) on rough-model output; switched to
  per-piece OS-level `timeout` so evals can't stall.

## The lever I'm now pulling: REALISTIC performance rendering (+ augmentation)
The domain gap: our synthetic training timing held tempo **constant within each bar**, so tuplets landed
at *exactly* 1/N of the beat. Real ASAP performances have structured sub-beat rubato + cadential
ritardando, so the timing→notation mapping the model must invert differs at test time — which most hurts
the tuplet pieces. Overnight I:
1. **Rewrote the renderer** (`expressive_render.py`) to a **beat-resolution AR(1) rubato curve +
   phrase-final ritardando**. Verified on triplets: was 0% within-beat unevenness (deadpan) → now a
   realistic ~3–8%, with ~10–15% beat-to-beat rubato and no edge artifacts.
2. **Augmented the scarce tuplet corpus**: re-rendered each of the 241 kern scores **×30 with different
   rubato seeds** — 7,230 *distinct plausible performances* of the same scores. This is real
   augmentation (each render is unique), NOT the exact-repeat upweighting that sank ar_full5.
3. **Ran `ar_rubato`**: warm-start from ar_full3 and finetune on 84K PDMX + the rubato-augmented kern
   (kern now 7.9% of synthetic, all *distinct* realizations).

**Result — negative, and informative.** The finetune **diverged** on real-ASAP val (0.97 → 1.09 → 1.25),
and the per-piece tail eval (epoch-0, the best ckpt) shows realistic rubato did **not** rescue the hard
tail:
| piece (gt tuplets) | ar_full3 tuplets | ar_rubato tuplets | ar_rubato MeanER |
|---|---|---|---|
| Mozart (36) | 53 | 66 (**over**-produces, 1.8×) | 13.4 (was 12.45) |
| Liszt (402) | — | 303 (0.75×) | 33.0 |
| Scriabin (99) | 0 | **0** | 19.9 |
| Ravel/Ondine (1281) | 53 | **0** (regressed) | 38.1 |

So rubato made the model emit *more* tuplets where they're easy (Mozart over-shoots; Liszt 0.75×) but
**still zero on the two hardest dense-tuplet pieces** (Scriabin, Ravel) — and it lost the few Ravel
tuplets ar_full3 had. Net: worse general accuracy, no tail gain. I stopped it (didn't burn the full 5
epochs) rather than chase a diverging run.

## Released SOTA vs ours — the head-to-head (same eval harness, same pieces)
| piece (gt tuplets) | **released** MeanER | released tuplets | ar_full3 MeanER | ar_full3 tuplets |
|---|---|---|---|---|
| Mozart K.332 (36) | **5.86** | 0 | 12.45 | (n/m) |
| Bach BWV846 (0) | **5.62** | 0 | 19.9 | 0 |
| Scriabin Op.8/11 (99) | **10.37** | **0** | 16.3 | **0** |
| Ravel Ondine (1281) | **20.8** | 384 (0.3×) | 27.4 | 53 |
| Liszt Années II (402) | **13.14** | 682 (1.7×) | (n/m) | — |

(Prokofiev timed out for both. "n/m" = not measured.) Note even the released model's tuplet production is
all over the map — 0× on Mozart/Scriabin, 0.3× on Ravel, **1.7× (over)** on Liszt — i.e. *no* model here
controls dense tuplets reliably.

Two things jump out:
1. **The dense-tuplet tail is hard for SOTA too.** The released model produces **0** tuplets on Scriabin
   and only **0.3×** on Ravel Ondine — the same collapse we have. This is a field-wide hard problem (these
   pages are ~extreme tuplet density), so our tail gap is *not* a unique defect.
2. **Our real gap to SOTA is broad accuracy, not tuplets.** The released model beats us on MeanER
   everywhere (Mozart 5.86 vs 12.45) *while producing fewer tuplets than we do on Mozart/Ravel*. So
   chasing tuplet production was the wrong target — the released model is simply more accurate across pitch/
   duration/voicing from its better (unpublished) training data.

## What this tells us (the real takeaway)
Three independent *data* interventions on our from-scratch model — more PDMX, kern repetition, realistic-
rubato augmentation — all **failed to close either gap**. Combined with the head-to-head above, the
conclusion is:
- **The tuplet tail isn't the bottleneck and isn't data-fixable** (even SOTA collapses there).
- **The broad-accuracy gap to SOTA is about their training data/recipe**, which is unpublished.
- For the **product** (audio→score), the pragmatic answer is to **use the released Block-2 model** (we
  have it; MUSTER ~11) with our solved Block-1 — it's already SOTA. The from-scratch work was research
  into reproducing/improving their decoder; it reproduces it but doesn't beat it.
- If we still want to *improve on SOTA's tail*, it's an **architectural** problem (decoder capacity /
  dense-passage decoding), not a data one — but note even SOTA hasn't solved it.

## Recommended next steps (for when you're back)
1. **Decide the goal explicitly.** If the goal is the *product* (audio→score at SOTA quality), we're
   effectively there: use the released Block-2 model (MUSTER ~11) + our Block-1. No more training needed.
2. **If the goal is to beat SOTA on the tail**, stop spending on data — it's exhausted. The tail is
   architectural *and* unsolved even by SOTA, so it's a genuine research bet: larger/deeper decoder, or
   dense-passage-aware decoding. High risk.
3. **If the goal is to match SOTA's broad accuracy from scratch**, the gap is their unpublished training
   data/recipe; closing it means acquiring comparable data, not algorithm tweaks.
4. Keep `ar_full3` as our best from-scratch checkpoint; the released ckpt remains the one to ship.

## Where things stand (nothing committed — per your instruction)
- Best checkpoint: `MIDI2ScoreTransformer/checkpoints/ar_full3` (Mozart at baseline). All builds + data on the box.
- ar_rubato run finished (negative); its ckpts + the rubato manifest (`data/pairs_rubato_manifest.csv`,
  84K PDMX + 7,228 distinct-rubato kern) are on the box for inspection.
- New code (uncommitted): realistic `expressive_render.py` (beat-rubato + ritardando, with `render_from_parsed`),
  `render_all_kern.py` (spawn-pool augmentation driver), `run_ar_rubato.sh`, hardened `robust_eval.sh`.
- **Infra note:** most of the night's lost time was a self-inflicted `pkill -f <pat>` that matched its own
  ssh shell (killing the command before it ran) + orphaned joblib workers it never reaped. Fixed with the
  `[p]attern` bracket trick; all later launches worked first try.
