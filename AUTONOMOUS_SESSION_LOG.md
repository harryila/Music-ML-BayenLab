# Autonomous GPU session — log (for the 10am report)

**Started:** 2026-06-07 (user asleep ~10h). **Box:** root@213.192.2.122:40070, 2× RTX 3090, 256 vCPU, 1007 GB RAM, repo `/root/Music-ML-BayenLab`.
**Goal:** close the +1.10 MeanER gap (best from-scratch `ssl_tuplet20`=11.87 → released SOTA=10.77 on the 14-piece ASAP test), ideally beyond. **Frontier:** tuplet PLACEMENT/DURATION (global knobs recover count but not placement → plateau at 11.87).

## Method discipline (per user: don't be lazy, double/triple-check)
- **Eval gate:** every new ckpt is scored by REAL MUSTER on the 14-piece set; nothing is trusted until the pipeline reproduces a KNOWN number (ssl_classical_clean Mozart=5.76, Scriabin=12.32; released Scriabin=10.37).
- **Selection by MUSTER, not val/total** (val≠MUSTER is proven).
- **Warm-start vehicle only** (from ssl_classical_clean ep13) — plain fine-tune / from-scratch drifts MUSTER.
- **A/B controls:** every lever has a gamma=0 / weight=1 identity check.
- **Discriminating design:** experiments are chosen to distinguish *why* tuplets collapsed (output-prior vs input-timing vs per-piece-placement), not just to chase the number.

## Hypotheses for the tuplet collapse (ours collapses, released doesn't; same arch + same pitch-only surrogate)
- H1 output-prior: our unpaired corpus is ~86% tuplet-free → decoder learns P(tuplet)≈0.
- H2 input-timing: pitch-only surrogate gives no timing → can't learn WHEN to place tuplets from unpaired data.
- H3 exposure/balance: real(822, tuplet-rich) vs unpaired(24k, tuplet-poor) mix under-weights the teaching signal.
- H4 placement: even with a correct prior, a single global model can't know a piece's tuplet density (the global-knob plateau).

## Experiment queue (warm-start, MUSTER-eval, 2 GPUs in parallel)
- E0  re-rank ALL existing epochs by MUSTER (free; may beat 11.87 already on disk)
- E1  output-prior rebalance: high-tuplet unpaired corpus (FIX the reshape no-op guard) [H1]
- E2  denoising-SSL: corrupt surrogate timing, reconstruct clean rhythm [H2]
- E3  per-piece tuplet-density conditioning token (non-dyadic IOI fraction) [H4]  ← most novel
- E4  duration-stream-only / focal / scheduled tuplet loss [refine the w=20 optimum]
- E5  exposure rebalance (raise real-pair fraction)
- LIT literature scan (2024-26 AMT→score / rhythm quantization / tuplets) to ground + improve the above

---

## RUNNING LOG
(appended as work proceeds)

### [Gate] Env bootstrap + eval reproduction — PASSED
- 2× RTX 3090, torch 2.12+cu130 sees both GPUs. Fixed: score_transformer needed `bs4`; **transformers 5.x breaks MyModelConfig → pinned transformers==4.44.2** (tokenizers 0.19.1).
- ASAP + ACPAS data set up; ckpts scp'd from local backup.
- **Reproduction (the gate):** ssl_classical_clean ep13 → Mozart **5.76** (doc 5.76 ✓), Scriabin **12.29** (doc 12.32 ✓). Pipeline trustworthy.

### [LIT] Literature scan — DONE → strategy pivot (research-grounded)
Convergent finding: tuplet mis-placement is caused by the **flat absolute 1/24 grid**; fixes are POSITIONAL/structural, not count-knobs. Saved `LIT_TUPLET_FINDINGS.json`. Recommended sequence (cheapest placement-true lever first):
- **A1 — logit-adjustment τ-sweep on duration logits** [zero retrain] (Menon ICLR'21): `logit_y -= τ·log π_y`. Fisher-consistent rebalance; shifts the decision boundary only where evidence nearly supports a tuplet → fixes placement, not just count. *Run first as a calibration probe.*
- **A2 — metrical-position duration prior** [precompute table, zero retrain] (Shibata 2021): add `λ·log P(duration | offset-position)` so tuplet durations are cheap only at triplet sub-beats {8,16}, expensive at binary {6,12,18}. Directly attacks placement.
- A3 grammar/qparse per-beat rescorer [decode-time]; B1 MiLe/token-adaptive duration loss [short fine-tune]; **B2 beat-relative retokenization in denoising-SSL [full retrain, highest leverage]** (Wachter 2025 — beats Beyer on onset).
- DEMOTE E1 (more tuplet-rich corpus = same failed global-prior family). Hard constraint (own memory): no beat-conditioning as an INPUT token on the absolute grid; beat info must reshape TOKENIZATION at pretrain.
- Report per-piece tuplet precision/recall, not just count/mean (prior failures recovered count while misplacing).
**Decision:** implement A1+A2 now (zero-retrain, breakthrough candidates, parallel the pad-threshold win), sweep on ssl_tuplet20 (11.87) by MUSTER; queue B1/B2 retrains for the GPUs.

### [Exec] Pipeline running (both GPUs busy)
- Training data READY: unpaired cache 23,775/23,783 rebuilt (256-core), ASAP chunks 967 ✓.
- A1/A2 levers implemented + locally validated (identity-off; A1 live). `compute_duration_priors.py` (ASAP train+val), `run_dur_prior_sweep.sh`. Files synced to box.
- GPU0: E0 re-rank by MUSTER (all saved epochs). GPU1: A1/A2 τ/λ sweep on ssl_tuplet20 (chained on prior). Poller watching both.
- Next: analyze E0 + A1/A2 → if inference levers break 11.87, refine; else B-stage retrains (B1 MiLe duration loss, B2 beat-relative retokenization).

### [A2 prior] VALIDATED — metrical prior is correct
Computed from 201 real engraved ASAP train+val scores (514,645 notes; overall tuplet_frac 11.3%). Sanity:
- phase 8 (triplet): top durations [8,4,16] = ALL tuplet ✓
- phase 16 (triplet): top durations [8,4,16] = ALL tuplet ✓
- phases 0/6/12/18 (binary): top durations dyadic ✓
So `P(duration|phase)` encodes "tuplets only at triplet sub-beats" — the placement signal. A2 design confirmed.

### [E0] MUSTER re-rank — DONE
Under THIS env (transformers 4.44.2), baselines shift ~0.09 vs old box; compare new results to these:
- ssl_tuplet20 **last = 11.79** (best on disk; beats its val-best epoch_01=12.15 → confirms val≠MUSTER)
- ssl_reshape_g1 ep00 = 12.01; ssl_classical_clean ep13 = 12.77 (env baseline). No hidden better epoch.

### [A1] logit-adjustment (global) — FAILED (as LIT predicted)
τ-sweep on ssl_tuplet20 (baseline 11.68 @0.50): τ=0.5→13.91 (tuplets 2954→6395, over-emit), τ=1.0→32.96 (25k tuplets), τ≥1.5 catastrophic. A global duration-prior shift over-emits everywhere = the same placement failure as loss-reweighting. **Confirms placement is NOT globally calibratable.**
→ A2 (POSITIONAL metrical prior) is the real test; finer λ curve running across both GPUs.

### [B2 scoping] beat-relative tokenization
- A beat-phase INPUT stream already exists (tokenizer.py:119-137, beat_features.py) — but that's the input-conditioning approach my memory records as neutral-to-destructive. The LIT lever is beat-relative OUTPUT tokenization (offset currently within-MEASURE, PARAMS offset 0..6 q @1/24; would need within-BEAT offset head + detokenize reconstruction) — a real refactor.
- DECISION RULE for B-stage (decide with data): wait for A2 curve; if A2 (positional inference prior) helps → positional axis confirmed, invest there; if A2 flat → run an offset-phase diagnostic on ssl_tuplet20's Scriabin prediction (are predicted offsets ever at triplet phases 8/16, or all binary?). If offsets are all-binary, A2/A3 can't help and beat-relative OUTPUT retokenization (B2) is the needed structural fix; else token-adaptive bounded duration loss (B1, Gu et al.) is the safer retrain.

### [A2] positional metrical prior — MODEST WIN (inference, zero-retrain)
A2 λ-curve on ssl_tuplet20 (baseline 11.676 @0.50): **λ=0.75 → 11.456 (−0.22)**, λ=0.25→11.54, λ=0.5→11.64; λ≥1.0 worse. A2 works by SUPPRESSING mis-placed tuplet durations (pred_tup 2954→902), not adding correct ones. New inference-best = **11.456** (A2 λ=0.75 on ssl_tuplet20).

### [DIAGNOSTIC] offset-phase — the root cause, and it reframes the strategy
ssl_tuplet20 on Scriabin: **PRED places 0/1408 notes at triplet offset phases (0.0%)**; GT has 66 (4.78%). The model quantizes EVERY offset to binary. So positional INFERENCE levers (A2/A3) can only clean up duration; they can't add tuplets (model never places triplet offsets). BUT our arch == released (which DOES place triplet offsets on the SAME grid) → the grid is NOT the blocker; the **tuplet-poor corpus (86% tuplet-free) collapsed the offset prior, and loss-reweighting (w=20) can't overcome it.** Lever = training data that teaches placement in-context.
→ B-stage: (B-real) raise real ASAP fraction 0.5→0.7 (clean triplet placement signal); (B-corpus) triplet-rate-balanced unpaired corpus via FIXED gamma reshape. Eval by MUSTER + re-run offset diagnostic (did triplet placement appear?).

### [B-stage] data experiments LAUNCHED (both GPUs, warm-start ssl_classical_clean + tuplet_weight 20)
Goal: teach triplet PLACEMENT (the diagnosed root cause: 0% triplet offsets).
- GPU0 **ssl_real07**: real-ASAP fraction 0.5→0.7 (lean on the 822 clean-triplet pairs). manifest=rebuilt.
- GPU1 **ssl_balanced**: triplet-rate reshape γ=1.5 on rebuilt corpus (tuplrate computed: γ1.0→9.1%, γ1.5≈~18%, γ2.0→33.6% effective vs ASAP-train ~11%). Tests corpus composition done right (the lever ssl_bigc botched by complexity-skew).
- Eval plan when done: 14-piece MUSTER @0.50 (+A2 λ=0.75) AND re-run offset-phase diagnostic (did triplet placement appear? — the real success metric, beyond MeanER).
- ROUND-2 contingency if neither induces triplet placement: (a) stronger γ (2.0) / higher real-frac (0.85); (b) beat-relative OUTPUT retokenization (B2, Wachter) — sample-efficient triplet learning, bigger refactor; (c) accept A2-inference win (11.456) as the deliverable.

### [B-stage] RESULT — placement collapse BROKEN via corpus composition 🎯
Offset-phase diagnostic on Scriabin (GT triplet-offset rate 4.78%):
- ssl_real07 (real-frac 0.7): **0.0% triplet offsets** — more real pairs did NOT teach placement.
- ssl_balanced (γ=1.5 triplet-rate reshape): **19.9% triplet offsets** (0% → 19.9%!). The triplet-rate-balanced corpus INDUCES triplet placement — the diagnosed root cause is fixable by corpus composition.
- γ=1.5 OVER-places (19.9% vs 4.78% GT) → likely over-emits → MUSTER may suffer (ssl_bigc pattern). Tuning γ down is round 2.
val: ssl_balanced ep00 0.5089 (beats baseline 0.5125). MUSTER re-rank running (GPU0) to see if placement→better MeanER.
→ ROUND 2: γ-sweep down (1.0, 0.5) to match the realistic ~5-11% rate; then + A2 inference lever for the final number.

### [B-stage round-1 MUSTER] — placement induced, but OVER-emits → MeanER regressed (honest)
14-piece MUSTER @0.50 (env baseline ssl_classical_clean=12.77, best ssl_tuplet20=11.79):
- ssl_real07 (real-frac0.7): 13.65 / 13.94 / 15.02 — REGRESSED (more real exposure hurt).
- ssl_balanced (γ1.5): 15.68 / 16.00 / 16.07 — REGRESSED hard. The 19.9% triplet placement OVER-emits (vs GT 4.78%) → wrecks MeanER, the ssl_bigc pattern.
- KEY: reshape runs have BETTER val (0.503-0.509) but WORSE MUSTER → val is anti-correlated here (fits tuplet-rich corpus, over-emits on real test). MUST judge by MUSTER + placement-rate.
→ The placement collapse is breakable (real finding) but at a MeanER cost; need placement at the RIGHT rate. Round-2 γ-sweep (1.0, 0.5, val 0.505/0.503) being MUSTER+diagnostic-eval'd. Idea: A2 (suppresses mis-placed tuplets) ON TOP of a reshape model may rescue the over-emission.

### [B-stage round-2 γ-sweep] — reshape is MUSTER-NEGATIVE at all γ (clean negative)
Placement rate (Scriabin, GT 4.78%) / MUSTER: γ0.5 → 0.0% / regressed; γ1.0 → 1.4% / 13.74; γ1.5 → 19.9% / 16.0.
- Sharp transition (γ1.0≈none → γ1.5 over-emits); NO γ gives correct-rate placement + good MeanER.
- Even γ1.0 (barely any placement) has MUSTER 13.74 > baseline 12.77 → the resampling DRIFTS the model (overweights the same tuplet-rich scores → overfits/over-applies). Reshape = softer ssl_bigc, same failure.
**Conclusion: corpus reshape proves placement is INDUCIBLE but does NOT translate to MeanER (over-emits/drifts).** The released edge is correct-RATE placement, which our data levers can't hit.
→ Testing the novel combo: A2 (suppress tuplet durations at binary offsets) ON the over-emitting γ1.5 model — keep correctly-placed triplets, remove mis-placed ones.

### [Direction: user chose BOTH beat-relative + self-training]
- A2-on-reshape: deprioritized/killed (partial; A2 on the over-emitter wasn't resolving it fast). GPUs freed for the two bigger tracks.
- **Track ST (RUNNING):** released teacher pseudo-labels 200 shortest MAESTRO performances → tuplet-CORRECT scores → added (×12 upweight) to the unpaired SSL corpus → warm-start retrain `ssl_pseudo` (tuplet_weight 5). Rationale: inject correctly-placed tuplets at NATURAL repertoire distribution → fix the tuplet-poor corpus that collapsed the offset prior, WITHOUT the reshape's over-emission. Fully chained (scripts/pseudo_label_maestro.py + build_and_train_st.sh). This is the data-side route to correct-rate placement.
- **Track B2 (beat-relative) — feasibility resolved + specced, deferred (not rushed):** faithful beat-relative needs a NEW beat-boundary output head (within-beat offset can't be inverted without a beat index; offset-decrease detection fails when consecutive beats share a phase). That's a retrain-from-modified-architecture + detokenize rewrite, requiring a lossless tokenize→detokenize round-trip gate BEFORE training. SPEC: (1) offset := offset_in_measure mod beatLength; (2) add `beat_idx` head (beat# within measure); (3) detokenize: offset_in_measure = beat_idx*beatLength + within_beat_offset; (4) round-trip test on ASAP scores must be lossless before any retrain. Not rushed overnight (high silent-corruption risk); ST covers the same objective tonight. To run in a focused session OR if ST frees GPUs with ample time.

### [A2-on-reshape] PARTIAL (killed for ST) — promising direction
A2 (λ=0.75) on the over-emitting reshape models cut tuplet over-emission massively: g1.0 pred_tup 10026→980, g1.5 12986→932. MeanER over the (partial) 6-7 pieces scored before kill = ~10.4 — NOT comparable to full-14 (easy pieces scored first). Signal: A2 *can* clean a placement-inducing model's over-emission. TODO: full-14 A2-on-reshape eval when a GPU frees (ST retrain uses only GPU0 → GPU1 free during that phase).

### [ST round-1 result] ssl_pseudo UNDER-places (0.9% triplet offsets) — too dilute
200 pseudo×12 (~9% of corpus) → placement 0%→0.9% only (vs reshape γ1.5 19.9% over; GT 4.78%). MUSTER pending.
KEY: reshape OVER-shoots, pseudo UNDER-shoots → correct-rate placement is a NARROW target on the absolute grid.
~8h remain (box clock ~01:30). PLAN: (a) ST round-2 — scale pseudo to ~600 DIVERSE MAESTRO pieces at moderate
weight (more correct-tuplet exposure without overfit-drift); (b) carefully attempt B2 beat-relative (new beat_idx
head + input embedding + detokenize + LOSSLESS round-trip gate) — the structural fix that makes correct-rate the
natural outcome. Both feasible in the remaining time.

### [FINAL data-lever verdict] all 4 retrains REGRESSED; A2 inference (11.456) is the durable win
ssl_pseudo 13.31 (over-emits durations 9242, under-places offsets 0.9%); A2-on-reshape 12.96; ssl_real07 13.65; reshape 13.7-16.0. NONE beat baseline 12.77 / best 11.79. Correct-rate placement unlearnable via data/loss on absolute grid. → committing remaining time to a careful, round-trip-gated B2 (beat-relative) attempt.

### [B2 implementation plan — file-by-file, for a focused session]
Surgery confirmed HIGH (hardcoded per-stream embeddings/heads): NOT rushed overnight (silent-corruption risk).
Concrete plan (precedent: the existing opt-in `beat` INPUT stream in embedding.py shows the add-a-stream pattern):
1. config.py: add out_beat_idx_vocab_size (e.g. 16 = max beats/measure) + in_beat_idx_vocab_size; gate behind use_beat_relative.
2. tokenizer.py bucket_mxl: beatLen = measure.barDuration.quarterLength / beatCount (or 1.0 for x/4); offset := (note_offset-measure_offset) % beatLen; new beat_idx := floor((note_offset-measure_offset)/beatLen). Re-bucket offset to within-beat range.
3. embedding.py MXLEmbeddings (decoder in) + the decoder OUTPUT heads: add beat_idx Linear (mirror an existing stream), opt-in + zero-init for clean warm-start.
4. train.py loss: add beat_idx CE (same machinery as other streams).
5. model.generate: sample beat_idx like other streams; feed back in AR.
6. tokenizer.detokenize_mxl: offset_in_measure = beat_idx*beatLen + within_beat_offset; rebuild measures from that.
7. GATE: tokenize_mxl->detokenize_mxl round-trip MUST reproduce note offsets losslessly on ASAP train scores BEFORE any retrain.
8. Retrain SSL warm-start (body loads; beat_idx head trains from zero) ~30-40 ep; eval MUSTER + offset diagnostic.
Hypothesis: within-beat offset makes triplet positions COMMON buckets (validated: triplets concentrate at phase 8/16) → correct-rate placement learnable from the tuplet-sparse corpus → potential to beat 10.77 (Wachter 2025 precedent).

### [B2 IMPLEMENTED + TRAINING] beat-relative tokenization — the literature's #1 lever
Phase 1 (tokenizer) + Phase 2 (model) fully implemented, gated behind use_beat_relative/--beat-relative, all local tests PASS, non-B2 byte-identical:
- offset re-expressed as WITHIN-QUARTER (25 buckets) + new quarter_idx head (25). Lossless round-trip (Δ=0 to 1/24 grid); 100% of triplet-position notes land in common within-quarter buckets {8,16} (the sample-efficiency benefit). Split done in bucket_mxl (load-time) → EXISTING cache works, no rebuild.
- Files: tokenizer.py (BEAT_RELATIVE flag, PARAMS_BR, bucket_mxl split, detokenize reconstruct), config.py (use_beat_relative, quarter_idx vocab, offset->25 override, FEATURES), embedding.py (quarter_idx in decoder-input + output-head ParameterDicts), model.py (quarter_idx in generate y_start_token), train.py (--beat-relative wired through fit/_new_model/load_pretrained_init; SHAPE-FILTERED warm-start). Tests: test_beat_relative_roundtrip.py, test_b2_model_smoke.py.
- Warm-start verified: loaded 198/205 params, fresh-init 7 (offset+quarter_idx heads); body + pitch/dur/voice/hand warm-start from ssl_classical_clean.
TRAINING (both GPUs, 20ep, warm-start ssl_classical_clean, real_frac0.5): ssl_b2 (tuplet_weight 1 = pure representation) GPU0; ssl_b2_tw3 (tuplet_weight 3 hedge) GPU1.
EVAL plan (consistent, per request): full-14 MUSTER --limit-per 1 + offset diagnostic (within-quarter triplet placement) vs baselines 11.79 / released 10.77. HYPOTHESIS: within-quarter representation lets the model learn correct-rate triplet placement from the sparse corpus → potential to beat the data-lever plateau.

### [B2 RESULT — honest] representation necessary but NOT sufficient on tuplet-poor corpus
ssl_b2 (tw1, 20ep warm-start): full-14 MUSTER 12.87 (≈ baseline 12.77, worse than ssl_tuplet20 11.79).
ssl_b2_tw3 over-emitted (predtup 9901, 14.16). Per-piece predtup LUMPY (Scriabin 3, Schubert 2157, most 0).
OFFSET DIAGNOSTIC (Scriabin): PRED 0.07% triplet phases (1/1408) vs GT 4.78% — SAME COLLAPSE as absolute grid.
val plateaued at ep1 (0.53): the reinit offset head immediately relearned the dyadic-collapse solution in the
within-quarter representation. CONCLUSION: making triplets COMMON BUCKETS doesn't induce placement when the
corpus has ~no triplet SIGNAL (1.7%). Representation = necessary-not-sufficient; DATA tuplet-poverty dominates
(consistent with Wachter's beat-relative win being on tuplet-RICH data). NEXT (running): ssl_b2_data = B2 +
tuplet-correct pseudo-MAESTRO data (merged corpus, 30ep, lr3e-4) — tests representation+signal TOGETHER.
