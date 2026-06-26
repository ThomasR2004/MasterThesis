import argparse
import json
import os
import numpy as np
from tqdm import tqdm

from load_data import load_train_data, load_test_data, load_train_embeddings, load_test_embeddings
from embed_data import get_closest_embeddings

_KEY_MAP = {
    "en": "English", "es": "Spanish", "ja": "Japanese",
    "zh": "Simplified Chinese", "fr": "French", "it": "Italian", "pt": "Portuguese",
}


_DATA_KEYS = {
    "politeness": {},  
    "formal":     {},  
    "intimacy": {"en": "English", "es": "Spanish", "pt": "Portuguese",
                 "it": "Italian", "fr": "French", "zh": "Chinese"},
}


def resolve_data_key(style, lang):
    """Map a CLI lang code to the key the LOADERS use for this style (per embed_data)."""
    return _DATA_KEYS.get(style, {}).get(lang, lang)


def resolve_target_levels(source_label, base_levels, schedule, include_source_label):
    """Native-scale target levels to train on (mirrors translate.py)."""
    L = float(source_label)
    lo_lvl, hi_lvl = float(min(base_levels)), float(max(base_levels))
    mid = (lo_lvl + hi_lvl) / 2.0
    if schedule == "closefar":
        far = hi_lvl if L < mid else lo_lvl
        levels = {round(L, 3), float(far), round(mid, 3)}
    else:
        levels = {round(b, 3) for b in base_levels}
    if include_source_label:
        levels.add(round(L, 3))
    return sorted(levels)


def _style_shift(style, level, source_key, target_key, embeddings, labels, cache):
    key = (source_key, target_key, round(float(level), 6))
    if key in cache:
        return cache[key]
    se, te = embeddings[source_key], embeddings[target_key]
    sl, tl = labels[source_key], labels[target_key]
    sdiff = [abs(x - level) for x in sl]
    tdiff = [abs(x - level) for x in tl]
    ns, nt = int(len(se) * 0.1), int(len(te) * 0.1)
    if style == "formality":
        ns = int(np.count_nonzero(np.array(sdiff) == 0))
        nt = int(np.count_nonzero(np.array(tdiff) == 0))
    si = np.argsort(sdiff)[:ns]
    ti = np.argsort(tdiff)[:nt]
    shift = np.mean(te[ti], axis=0) - np.mean(se[si], axis=0)
    cache[key] = shift
    return shift


def align_embedding(style, embedding, level, source_key, target_key, embeddings, labels, cache):
    return embedding + _style_shift(style, level, source_key, target_key, embeddings, labels, cache)


def load_prompt(style, source_key, target_key, text, level, rag_examples):
    template_map = {
        "politeness": "prompts/politeness_translation.txt",
        "intimacy":   "prompts/intimacy_translation.txt",
        "formal":     "prompts/formality_translation.txt",
    }
    if style in ("politeness", "formal", "intimacy"):
        source_key = _KEY_MAP.get(source_key, source_key)
        target_key = _KEY_MAP.get(target_key, target_key)
    with open(template_map[style]) as fh:
        prompt = fh.read()
    return prompt.format(
        source_key, target_key, text, level,
        source_key, target_key,
        rag_examples[0], rag_examples[1], rag_examples[2], rag_examples[3], rag_examples[4],
        level, target_key,
    )


def get_rag_examples(style, target_key, data, embeddings, labels, aligned, level, exclude_index):
    tgt_data, tgt_emb, tgt_lab = data[target_key], embeddings[target_key], labels[target_key]
    if exclude_index is not None and 0 <= exclude_index < len(tgt_data):
        tgt_data = [s for j, s in enumerate(tgt_data) if j != exclude_index]
        tgt_emb = np.delete(tgt_emb, exclude_index, axis=0)
        tgt_lab = [l for j, l in enumerate(tgt_lab) if j != exclude_index]
    ex, _, _ = get_closest_embeddings(style, tgt_data, tgt_emb, tgt_lab, aligned, level, 5)
    return ex


def ensure_k_examples(examples, k, on_short, level, starved):

    examples = list(examples)
    n = len(examples)
    if n >= k:
        return examples[:k]
    if n == 0:
        # Nothing to cycle from -- padding would be meaningless.
        raise ValueError(
            f"retrieval returned 0 exemplars at level {level} "
            f"(empty target pool after exclude); cannot build prompt."
        )
    starved.append((round(float(level), 3), n))
    if on_short == "error":
        raise IndexError(
            f"retrieval returned {n} exemplars (<{k}) at level {level}; "
            f"template needs exactly {k}. Use --rag_short pad|skip to handle this."
        )
    if on_short == "skip":
        return None
    # "pad": cycle deterministically up to k.
    return [examples[i % n] for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="politeness")
    ap.add_argument("--src_lang", required=True, help="e.g. en")
    ap.add_argument("--tgt_lang", required=True, help="e.g. zh")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--target_levels", default="0,1,2,3,4",
                    help="Native-scale grid (prompt is 'out of 4').")
    ap.add_argument("--schedule", choices=["grid", "closefar"], default="grid")
    ap.add_argument("--no_include_source_label", dest="include_source_label",
                    action="store_false")
    ap.set_defaults(include_source_label=True)
    ap.add_argument("--label_min", type=float, default=0.0,
                    help="Native-scale MIN (politeness/formal=0; a 1-based intimacy "
                         "scale would be 1). Used for the affine [0,1] mapping.")
    ap.add_argument("--label_scale", type=float, default=4.0,
                    help="Native-scale MAX. Stored target = "
                         "(native level - label_min) / (label_scale - label_min) -> [0,1].")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--rag_k", type=int, default=5,
                    help="Exemplars per prompt; must match the template's example slots.")
    ap.add_argument("--rag_short", choices=["pad", "skip", "error"], default="pad",
                    help="When retrieval returns < rag_k: pad (cycle), skip the row, "
                         "or error out.")
    ap.add_argument("--output", required=True, help="Output JSONL path.")
    args = ap.parse_args()

    base_levels = [float(x) for x in args.target_levels.split(",") if x.strip()]

    if args.label_scale <= args.label_min:
        raise ValueError(
            f"--label_scale ({args.label_scale}) must be > --label_min "
            f"({args.label_min}); these are the native MAX and MIN."
        )
    span = args.label_scale - args.label_min
    out_of_range = [lvl for lvl in base_levels
                    if lvl < args.label_min - 0.334 or lvl > args.label_scale + 0.334]
    if out_of_range:
        raise ValueError(
            f"--target_levels {out_of_range} fall outside the native range "
            f"[{args.label_min}, {args.label_scale}] for style {args.style!r}. "
            f"Retrieval keys on label proximity, so these levels would yield no "
            f"exemplars. Set --target_levels / --label_min / --label_scale to this "
            f"style's actual scale (e.g. intimacy on 1..5 -> --target_levels 1,2,3,4,5 "
            f"--label_min 1 --label_scale 5)."
        )

    if args.split == "train":
        data, labels = load_train_data(args.style)
        emb = load_train_embeddings(args.style)
        rag_data, rag_emb, rag_lab = data, emb, labels
    else:
        data, labels = load_test_data(args.style)
        emb = load_test_embeddings(args.style)
        tr_data, tr_lab = load_train_data(args.style)
        rag_data, rag_emb, rag_lab = tr_data, load_train_embeddings(args.style), tr_lab

    # Alignment stats always come from TRAIN (as in translate.py).
    tr_data2, tr_labels2 = load_train_data(args.style)
    tr_emb2 = load_train_embeddings(args.style)

    s, t = args.src_lang, args.tgt_lang
    s_key = resolve_data_key(args.style, s)
    t_key = resolve_data_key(args.style, t)

    if s_key not in data:
        raise KeyError(
            f"source key {s_key!r} (from --src_lang {s!r}, style {args.style!r}) "
            f"not in loaded data. Available keys: {list(data.keys())}"
        )
    if t_key not in data:
        raise KeyError(
            f"target key {t_key!r} (from --tgt_lang {t!r}, style {args.style!r}) "
            f"not in loaded data. Available keys: {list(data.keys())}"
        )

    n = len(data[s_key])
    if args.max_samples is not None:
        n = min(n, args.max_samples)

    tgt_pool_len = len(rag_data[t_key])
    parallel = (args.split == "train") and (len(data[s_key]) == tgt_pool_len)
    if args.split == "train" and not parallel:
        print(f"[warn] source {s}={len(data[s_key])} and target {t}={tgt_pool_len} "
              f"pools are not equal-length; treating this pair as non-parallel and "
              f"disabling index-based leave-one-out (no reference leakage to remove).")

    shift_cache = {}
    starved = []
    skipped_rows = 0
    rows = []
    for i in tqdm(range(n)):
        text = data[s_key][i]
        label = float(labels[s_key][i])
        embedding = emb[s_key][i]
        exclude = i if parallel else None

        for lvl in resolve_target_levels(label, base_levels, args.schedule,
                                         args.include_source_label):
            aligned = align_embedding(args.style, embedding, lvl, s_key, t_key,
                                      tr_emb2, tr_labels2, shift_cache)
            rag = get_rag_examples(args.style, t_key, rag_data, rag_emb, rag_lab,
                                   aligned, lvl, exclude)
            rag = ensure_k_examples(rag, args.rag_k, args.rag_short, lvl, starved)
            if rag is None:          # --rag_short skip
                skipped_rows += 1
                continue
            prompt = load_prompt(args.style, s, t, text, round(lvl, 3), rag)
            rows.append({
                "prompt": [{"role": "user", "content": prompt}],
                "target": min(1.0, max(0.0,
                              (float(lvl) - args.label_min) / span)),
                "source_text": text,
                "target_native": round(float(lvl), 3),
            })
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    if starved:
        from collections import Counter
        by_level = Counter(lvl for lvl, _ in starved)
        print(f"[warn] {len(starved)} (sentence,level) pairs had < {args.rag_k} "
              f"exemplars (policy: {args.rag_short}).")
        print(f"[warn] starved counts by level: {dict(sorted(by_level.items()))}")
        if args.rag_short == "skip":
            print(f"[warn] dropped {skipped_rows} rows due to starvation.")
    print(f"Wrote {len(rows)} GRPO prompts ({n} sentences x targets) -> {args.output}")


if __name__ == "__main__":
    main()