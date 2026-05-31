"""Tier-1 accuracy eval: MIDI -> score, scored with MUSTER + score_similarity
over the ASAP held-out test split (the field-standard, paper-comparable metrics).

This is the headline eval from ACCURACY_ROADMAP.md Track A. It runs any checkpoint
(default: the released MIDI2ScoreTF.ckpt) on the 14-piece / 59-performance ASAP
test split and reports the same numbers the Beyer ISMIR-2024 paper reports
(PitchER / MissRate / ExtraRate / OnsetER / OffsetER / MeanER, plus the notation
error rates), so we can anchor every future change to a real baseline.

Forces CPU: MPS produces degenerate pad logits on this model (documented in
PROJECT_REPORT.md), so the released checkpoint must run on CPU here.

Usage:
    venv311/bin/python benchmark/eval_tier1_asap.py \
        --ckpt MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt \
        --out benchmark/tier1_baseline.json
    # smoke test on the 3 shortest performances:
    ...  --limit 3
"""
import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

TF_ROOT = Path("/Users/harry/Desktop/temp/musicML/MIDI2ScoreTransformer")
sys.path.insert(0, str(TF_ROOT / "midi2scoretransformer"))
# ASAPDataset reads "./data/..." relatively, and {ASAP} paths resolve under cwd.
os.chdir(TF_ROOT)

import torch  # noqa: E402

# Same checkpoint-load incantation the production CLI uses (PyTorch 2.6 safe-globals).
from config import MyModelConfig  # noqa: E402
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])

from dataset import ASAPDataset  # noqa: E402
from models.roformer import Roformer  # noqa: E402
from tokenizer import MultistreamTokenizer  # noqa: E402
from utils import infer, eval as eval_pair  # noqa: E402


def load_any_checkpoint(ckpt_path, device):
    """Load either the released base Roformer or a fine-tuned TrainableRoformer
    checkpoint (train.py saves the latter). Tries base first, falls back."""
    try:
        return Roformer.load_from_checkpoint(ckpt_path, map_location=device, weights_only=False)
    except Exception:
        from train import TrainableRoformer  # noqa: E402
        return TrainableRoformer.load_from_checkpoint(ckpt_path, map_location=device, weights_only=False)

# Metrics we headline (match run_eval.py / the paper's "Ours" row).
MUSTER_KEYS = ["PitchER", "MissRate", "ExtraRate", "OnsetER", "OffsetER", "MeanER"]
NOTATION_KEYS = ["NoteDeletion", "NoteInsertion", "NoteDuration",
                 "StaffAssignment", "StemDirection", "NoteSpelling"]
PAPER_SOTA = {  # Beyer & Dai 2024, MUSTER (lower=better)
    "PitchER": 3.11, "MissRate": 7.56, "ExtraRate": 6.44,
    "OnsetER": 15.55, "OffsetER": 23.84, "MeanER": 11.30,
}


def collect_paths(split: str):
    q = ASAPDataset("./data/", split)
    md = q.metadata
    paths = []
    for i in range(len(md)):
        s = md.iloc[i]
        midi = s["performance_MIDI_external"].replace("{ASAP}", "./data/asap-dataset")
        score = os.path.dirname(midi) + "/xml_score.musicxml"
        rel = os.path.relpath(os.path.dirname(midi), "./data/asap-dataset")
        paths.append({"composer": s["composer"], "piece": rel,
                      "piece_id": int(s["piece_id"]), "midi": midi, "score": score})
    return paths


def aggregate(results):
    """Mirror run_eval.py aggregation: mean over non-None values per sub-metric."""
    done = [r for r in results if r.get("sim")]
    out = {"n_scored": len(done), "n_total": len(results)}
    for group in ("muster", "mxl <-> gt_mxl"):
        vals = {}
        for r in done:
            g = r["sim"].get(group) or {}
            for k, v in g.items():
                if isinstance(v, (int, float)):
                    vals.setdefault(k, []).append(v)
        out[group] = {k: (sum(v) / len(v) if v else None) for k, v in vals.items()}
    return out


def print_summary(agg):
    m = agg.get("muster", {})
    n = agg.get("mxl <-> gt_mxl", {})
    print("\n" + "=" * 72)
    print(f"TIER-1 (MIDI->score) on ASAP test split — {agg['n_scored']}/{agg['n_total']} scored")
    print("=" * 72)
    print("MUSTER (lower=better):       " + "  ".join(f"{k}" for k in MUSTER_KEYS))
    print("  Ours      ", "  ".join(f"{(m.get(k) or float('nan')):6.2f}" for k in MUSTER_KEYS))
    print("  Paper SOTA", "  ".join(f"{PAPER_SOTA[k]:6.2f}" for k in MUSTER_KEYS))
    print("\nNotation error rates (x100, lower=better):")
    for k in NOTATION_KEYS:
        v = n.get(k)
        print(f"  {k:16s} {100*v:6.2f}" if isinstance(v, (int, float)) else f"  {k:16s}   n/a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/MIDI2ScoreTF.ckpt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the N shortest performances (smoke test)")
    ap.add_argument("--overlap", type=int, default=64)   # match run_eval/paper
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--device", default="cpu",
                    help="cpu (default; required on Mac — MPS breaks this model) "
                         "or cuda (use on a GPU box for ~10x faster eval).")
    args = ap.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path("/Users/harry/Desktop/temp/musicML") / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    paths = collect_paths(args.split)

    # tokenize up front so we can sort by length (cheap; lets --limit pick short ones)
    print(f"Tokenizing {len(paths)} performances ...", flush=True)
    for p in paths:
        try:
            x = MultistreamTokenizer.tokenize_midi(p["midi"])
            p["_x"] = x
            p["n_notes"] = int(x["pitch"].shape[0])
        except Exception as e:
            p["_x"] = None
            p["n_notes"] = 1 << 30
            p["tok_error"] = str(e)
    paths.sort(key=lambda p: p["n_notes"])
    if args.limit:
        paths = paths[:args.limit]

    print(f"Loading checkpoint {args.ckpt} on CPU ...", flush=True)
    model = load_any_checkpoint(args.ckpt, device)
    model.eval()
    model.to(device)

    results = []
    meta = {"ckpt": str(args.ckpt), "split": args.split, "overlap": args.overlap,
            "chunk": args.chunk, "device": device, "n_performances": len(paths)}
    t0 = time.time()
    for i, p in enumerate(paths):
        rec = {k: p[k] for k in ("composer", "piece", "piece_id", "midi", "score", "n_notes")}
        if p["_x"] is None:
            rec["error"] = "tokenize: " + p.get("tok_error", "?")
            results.append(rec)
            continue
        try:
            ti = time.time()
            y_hat = infer(p["_x"], model, overlap=args.overlap, chunk=args.chunk,
                          verbose=False, kv_cache=True)
            sim = eval_pair(y_hat, p["score"])
            rec["sim"] = sim
            rec["infer_s"] = round(time.time() - ti, 1)
            mer = (sim.get("muster") or {}).get("MeanER")
            print(f"[{i+1}/{len(paths)}] {p['composer']:11s} {p['piece'][:34]:34s} "
                  f"notes={p['n_notes']:5d} MeanER={mer if mer is None else round(mer,2)} "
                  f"({rec['infer_s']}s)", flush=True)
        except Exception:
            rec["error"] = traceback.format_exc().splitlines()[-1]
            print(f"[{i+1}/{len(paths)}] {p['composer']:11s} {p['piece'][:34]:34s} ERROR: {rec['error']}", flush=True)
        results.append(rec)
        # incremental save so partial results survive a long/interrupted run
        agg = aggregate(results)
        json.dump({"meta": {**meta, "elapsed_s": round(time.time() - t0, 1)},
                   "aggregate": agg, "per_performance": results},
                  open(out_path, "w"), indent=2)

    agg = aggregate(results)
    json.dump({"meta": {**meta, "elapsed_s": round(time.time() - t0, 1)},
               "aggregate": agg, "per_performance": results},
              open(out_path, "w"), indent=2)
    print_summary(agg)
    print(f"\nWrote {out_path}  ({round(time.time()-t0,1)}s total)")


if __name__ == "__main__":
    main()
