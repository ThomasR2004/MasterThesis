import argparse
import glob
import os
import pickle
import re
import traceback

import numpy as np
import pandas as pd

from load_data import load_test_data, load_train_data
from comet import download_model, load_from_checkpoint



def cometkiwi(source_segs, target_segs, model, batch_size=8, gpus=1):
    data = [{"src": s, "mt": t} for s, t in zip(source_segs, target_segs)]
    out = model.predict(data, batch_size=batch_size, gpus=gpus)
    return out.scores


_INTIMACY_LANGS = {"English", "Spanish", "Portuguese", "Italian", "French", "Chinese"}


def split_from_name(fname):
    m = re.search(r'_(test|train)(?:_\d+)?\.pkl$', fname)
    return m.group(1) if m else "test"


def style_from_keys(translations):
    langs = set()
    for k in translations.keys():
        if "->" not in str(k):
            continue
        for part in str(k).split("->"):
            langs.add(part.strip())
    if not langs:
        return None, f"langs={sorted(langs)}"
    if langs & _INTIMACY_LANGS:
        return "intimacy", f"langs={sorted(langs)}"
    if langs & {"es", "ja", "zh"}:
        return "politeness", f"langs={sorted(langs)}"
    if langs & {"fr", "it", "pt"}:
        return "formal", f"langs={sorted(langs)}"
    return None, f"langs={sorted(langs)}"


def unwrap_translations(obj):
    source_from_file = {}
    if isinstance(obj, dict) and "translations" in obj and isinstance(obj["translations"], dict):
        inner = obj["translations"]
    elif isinstance(obj, dict):
        inner = obj
    else:
        return None, None

    # v2 detection (rows are dicts carrying their own source)
    for k, v in inner.items():
        if isinstance(v, list) and len(v) and isinstance(v[0], dict) \
           and "source_text" in v[0] and "candidates" in v[0]:
            translations = {}
            for key, rows in inner.items():
                translations[key] = [
                    r["candidates"] if isinstance(r["candidates"], list) else [r["candidates"]]
                    for r in rows
                ]
                source_from_file[key] = [r["source_text"] for r in rows]
            return translations, source_from_file
        break
    return inner, None  # v1: source must be re-loaded


def load_source_split(style, split):
    if split == "train":
        data, _ = load_train_data(style)
    else:
        data, _ = load_test_data(style)
    return data


def is_multi_candidate(raw):
    return len(raw) > 0 and isinstance(raw[0], list)


# ----------------------------------------------------------------------
# Score one pkl
# ----------------------------------------------------------------------
def run_one(pkl_path, comet_model, args):
    fname = os.path.basename(pkl_path)
    print(f"\n################ {fname}")

    obj = pickle.load(open(pkl_path, "rb"))
    translations, source_from_file = unwrap_translations(obj)
    if translations is None or not isinstance(translations, dict) or not translations:
        print("  !! not a translation dict; SKIPPING.")
        return (pkl_path, "SKIPPED (format?)", {})

    style, info = style_from_keys(translations)
    split = split_from_name(fname)
    print(f"  detected: style={style}, split={split}  ({info})")

    if style is None:
        print("  !! could not infer style; SKIPPING.")
        return (pkl_path, "SKIPPED (style?)", {})
    if args.style != "all" and args.style != style:
        print(f"  (filter) --style={args.style}; skipping this {style} file.")
        return (pkl_path, f"SKIPPED (filtered: {style})", {})

    data = None if source_from_file is not None else load_source_split(style, split)

    per_key_scores = {}
    per_key_mean = {}

    for key in translations.keys():
        lang1 = key.split("->")[0]
        raw = translations[key]
        multi = is_multi_candidate(raw)
        n_cand = len(raw[0]) if multi else 1

        if source_from_file is not None:
            src_all = source_from_file[key]
        else:
            if lang1 not in data:
                print(f"  [{key}] source lang {lang1!r} not in split (have {list(data.keys())}); skipping key.")
                continue
            src_all = data[lang1]

        n = min(len(src_all), len(raw))
        if len(src_all) != len(raw):
            print(f"  [{key}] LENGTH MISMATCH src={len(src_all)} vs mt={len(raw)} "
                  f"-> truncating to {n} (check the split is right!)")
        src = [str(s) for s in src_all[:n]]
        raw = raw[:n]

        print(f"  [{key}] scoring {n} segments x {n_cand} candidate(s)...")

        cand_score_cols = []
        for c in range(n_cand):
            mt = [row[c] for row in raw] if multi else list(raw)
            if not args.keep_full_text:
                # Match the style eval: take the first line as the translation.
                mt = [str(t).split("\n")[0] for t in mt]
            scores = cometkiwi(src, mt, comet_model,
                               batch_size=args.batch_size, gpus=args.gpus)
            cand_score_cols.append(scores)

        # (n_cand, n) -> (n, n_cand)
        per_item = [list(row) for row in zip(*cand_score_cols)]
        per_key_scores[key] = per_item

        cand0 = np.array([row[0] for row in per_item], dtype=float)
        per_key_mean[key] = float(cand0.mean())
        print(f"  [{key}] mean CometKiwi (cand 0) = {per_key_mean[key]:.4f}")

    # ---- save, named off the input stem so models stay distinguishable ----
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(fname)[0]
    out_path = os.path.join(args.out_dir, f"{stem}_cometkiwi.pkl")
    pickle.dump(
        {"style": style, "split": split, "scores": per_key_scores, "mean": per_key_mean},
        open(out_path, "wb"),
    )
    print(f"  saved -> {out_path}")

    overall = (float(np.mean(list(per_key_mean.values()))) if per_key_mean else float("nan"))
    return (pkl_path, f"OK (mean={overall:.4f})", per_key_mean)


def main():
    parser = argparse.ArgumentParser(description="Batch CometKiwi QE over translation pkls")
    parser.add_argument("--translations_dir", type=str,
                        default="/gpfs/home2/tvarelanunes/llm-translate/MasterThesis/pipeline/translations")
    parser.add_argument("--pkl_path", type=str, default=None,
                        help="Score a single pkl instead of the whole folder.")
    parser.add_argument("--out_dir", type=str,
                        default="/gpfs/home2/tvarelanunes/llm-translate/MasterThesis/pipeline/data/test_data/cometkiwi")
    parser.add_argument("--comet_model", type=str, default="Unbabel/wmt22-cometkiwi-da",
                        help="HF id of the QE model. Use Unbabel/wmt23-cometkiwi-da-xxl for the larger one.")
    parser.add_argument("--style", type=str, default="all",
                        choices=["politeness", "intimacy", "formal", "all"],
                        help="Filter to one style; 'all' scores every file.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--keep_full_text", action="store_true",
                        help="Score the full translation text instead of just its first line.")
    args = parser.parse_args()

    print(f"[comet] downloading / loading {args.comet_model} ...")
    model_path = download_model(args.comet_model)
    comet_model = load_from_checkpoint(model_path)
    print("[comet] model ready.")

    summary = []
    if args.pkl_path:
        summary.append(run_one(args.pkl_path, comet_model, args))
    else:
        pkls = sorted(glob.glob(os.path.join(args.translations_dir, "*.pkl")))
        pkls = [p for p in pkls if not p.endswith("_cometkiwi.pkl")]
        print(f"[batch] Found {len(pkls)} translation pkls in {args.translations_dir}")
        for p in pkls:
            try:
                summary.append(run_one(p, comet_model, args))
            except Exception as e:
                traceback.print_exc()
                summary.append((p, f"FAILED ({e})", {}))

    print("\n================ COMETKIWI SUMMARY ================")
    for p, status, per_key in summary:
        print(f"  {status:22s} {os.path.basename(p)}")
        for key, m in per_key.items():
            print(f"        {key:22s} {m:.4f}")


if __name__ == "__main__":
    main()
