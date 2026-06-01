# Session Log — 2026-05-31 → 06-01 (autonomous window)

Verified ground truth only. (An earlier attempt to write this log, and several
GPU/code actions, were cancelled mid-flight by a parallel-command cascade; this
reflects what actually completed, re-checked command by command.)

## The headline result
**The first real GPU fine-tune (synthetic-only) FAILED — it degraded MUSTER from
11.18 to 18.10.** A clean, decisive negative result. CUDA-baseline re-scored at
**11.10** on the full 59-piece split, so the device is not the confound; the +6.9
regression is real. Documented in
[benchmark/GPU_FINETUNE_RESULTS.md](benchmark/GPU_FINETUNE_RESULTS.md).

## Root cause (important): partly a tooling artifact
`train.py` had **no real-data loader** — `fit()` routed everything through the
synthetic-only `make_pdmx_loaders`. So "naive synthetic fine-tune" was the *only*
thing the code could do. With ~52k synthetic pairs vs ~750 real ASAP pairs (~65:1),
the synthetic data dominated and pulled the model off the real-performance
distribution it was tuned on.

## What's DONE (verified)
- **GPU findings doc** ([benchmark/GPU_FINETUNE_RESULTS.md](benchmark/GPU_FINETUNE_RESULTS.md)) —
  result table, diagnosis, operational lesson (the synthetic dataset is CPU-IO-bound;
  run ONE training at a time).
- **train.py tooling fix** — added `make_asap_loaders` (real ASAP) + `make_mixed_loaders`
  (80/20 real:synth WeightedRandomSampler replay, validates on real ASAP only) + a
  `--dataset-type {pdmx,asap,mixed}` / `--real-fraction` / `--data-dir` CLI. Syntax
  verified; 13 references present.
- **Repo hygiene** — the prior GPU run had committed ~275k synthetic-pair/cache files
  to git. Fixed `.gitignore`; **staged 275,243 deletions** via
  `git update-index --force-remove`. Files safe on disk. **NOT committed.**
- **Two research workflows** (verified, with sources):
  1. *Rendering landscape* — Google's Music Transformer / Performance RNN are
     *generators*, not score→performance renderers (can't make aligned pairs).
     VirtuosoNet is the best practical learned renderer (RenCon 2025 winner), but
     **no published work shows a renderer improves a MIDI→score model** — unproven bet.
  2. *Real-data path* — **PianoCoRe is CC BY-NC-SA (non-commercial)** → research-only
     for valency.io. Usable real data: **more ASAP** (already have, biggest easy win)
     and **ATEPP MusicXML half** (CC BY 4.0, commercial-OK, weeks of curation). ACPAS
     ships MIDI scores, not engraved MusicXML — not a new target source.

## Update (06-01): chunker fixed, ARM-1 LAUNCHED
- **Found + fixed a silent chunker bug** (path-prefix `KeyError` that skipped every
  un-chunked performance). Rebuilt 505 → **967** ASAP chunks. Details in
  GPU_FINETUNE_RESULTS.md.
- **Drift trajectory confirmed:** degradation is immediate (epoch 1 ≈ 17.7, final
  18.10), not gradual → it's the data mix, not over-training. So the fix is keeping
  real data in the loss (the mixed loader), not a gentler synthetic-only run.
- **Mixed/ASAP loaders verified on the GPU box** (real=822, synth=47612, 80/20
  sampler, validates on real ASAP=86). Shipped train.py + chunker.py + 967 chunks.
- **ARM-1 is now TRAINING** on the GPU: warm-start from released ckpt → mixed 80/20
  real:synth, lr 3e-5, 8 epochs, single-GPU (no DDP). In dataloader warmup → first
  steps. Monitor armed; will MUSTER-eval each saved checkpoint vs 11.18.

## Honest expectation for ARM-1 (when it runs)
Most likely an informative result near baseline (MeanER ~10.8–12.0) — a small win or
clean null — NOT a blowout. The released model is already well-trained on ASAP, so
limited headroom; and the synthetic inputs are noisy. One promising signal: even in
the failed run, `NoteDuration` *improved* (56.4→48.1), so the synthetic data carries
real engraving signal a proper mix might harvest without the timing drift.

## If ARM-1 fails (decision tree)
1. MUSTER ≥ 11.18 but generation healthy → lower synth share (90/10 or ASAP-only),
   or drop LR to 1.5e-5.
2. Try **ARM-2**: real paired + *unpaired* deduped PDMX scores at 50/50 via the
   unconditional branch — the **paper-faithful** path (the paper used unpaired scores,
   NOT synthetic pairs), likely higher ceiling. Needs a bit more plumbing.
3. If ARM-2 + ASAP-only both null → the verified bottleneck is genuinely-new REAL data
   (ATEPP MusicXML for commercial; PianoCoRe research-only). Stop tuning synthetic ratios.

## Files changed this session (nothing committed/pushed)
- **Modified:** `train.py` (loaders + CLI), `.gitignore` (bloat dirs),
  `benchmark/eval_tier1_asap.py` (robust loader + `--device`; also touched by you/linter),
  `transcribe.py` (Track B flags, earlier turn).
- **New docs:** `benchmark/GPU_FINETUNE_RESULTS.md`, this log.
- **Staged for git removal:** `data/pairs_deduped_*`, `data/cache_pdmx_*` (~275k files; on disk).

## GPU status
**Still in use** — drift evals running; ARM-1 to follow. Will signal explicitly when
the GPU is no longer needed.

## Open decisions for you
1. **Commit/push?** The git cleanup + new docs + train.py loaders are staged/changed
   but not committed. Say the word and I'll commit on a branch.
2. **Commercial vs research** decides whether PianoCoRe (non-commercial) is usable if
   ARM-1/ARM-2 null out.
