from load_data import load_test_data, load_test_embeddings, load_train_data
import argparse
import pickle
import gc
import glob
import re
import traceback
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Sequence Classification
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import torch
from transformers import BitsAndBytesConfig
from peft import PeftModel


def our_metric(source, target):
    return pearsonr(source, target).correlation



def normalize_labels(label_list, name=""):
    arr = np.asarray(label_list, dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if lo >= -1e-6 and hi <= 1.0 + 1e-6:
        # already in [0,1]; leave as-is
        return arr.tolist(), False
    if hi - lo < 1e-12:
        # constant labels: cannot min-max; return zeros and warn
        print(f"  [normalize] WARNING: labels for {name} are constant ({lo}); "
              f"correlation will be undefined.")
        return np.zeros_like(arr).tolist(), True
    scaled = (arr - lo) / (hi - lo)
    print(f"  [normalize] {name}: rescaled from [{lo:.3f}, {hi:.3f}] -> [0,1]")
    return scaled.tolist(), True



class AlignmentError(Exception):
    pass


def verify_alignment(key, candidates, source_texts, source_labels, strict=True,
                     expected_source_texts=None):
    """Check that the three lists can be safely zipped, and print samples."""
    n_c, n_t, n_l = len(candidates), len(source_texts), len(source_labels)
    print(f"  [align] lengths: candidates={n_c}, source_texts={n_t}, source_labels={n_l}")

    if not (n_c == n_t == n_l):
        lengths = {
            "candidates": n_c,
            "source_texts": n_t,
            "source_labels": n_l,
        }

        n = min(lengths.values())

        msg = (f"[align] LENGTH MISMATCH for {key}: "
               f"candidates={n_c}, source_texts={n_t}, source_labels={n_l}")

        # Auto-align to shortest list
        print(f"  {msg}")
        print(f"  [align] Auto-truncating all inputs to first {n} items")

        candidates = candidates[:n]
        source_texts = source_texts[:n]
        source_labels = source_labels[:n]

        # Refresh lengths after truncation
        n_c = n_t = n_l = n

    if expected_source_texts is not None:
        k = min(20, len(source_texts), len(expected_source_texts))
        matches = sum(
            1 for i in range(k)
            if str(source_texts[i]).strip() == str(expected_source_texts[i]).strip()
        )
    for i in range(min(3, len(source_texts))):
        cand = candidates[i][0] if isinstance(candidates[i], list) else candidates[i]
        print(f"    item {i} | src: {str(source_texts[i])[:70]!r}")
        print(f"           | tgt: {str(cand)[:70]!r}")
    return candidates, source_texts, source_labels


def get_style_model(style, lang, dir="/workspace/"):
    # 1. Initialize tokenizer first to capture correct pad token mappings
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 2. Load Sequence Classification base model with explicit pad_token_id
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        num_labels=1,
        pad_token_id=tokenizer.pad_token_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    adapter_path = f"{dir}training/{lang}/{style}_lora"
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"style classifier not found: {adapter_path} "
            f"(expected training/<lang>/<style>_lora; for intimacy <lang> is a full "
            f"name like 'Chinese', for politeness/formal it is a short code like 'zh')")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return model, tokenizer


_STYLE_MODEL_CACHE = {"key": None, "model": None, "tokenizer": None}


def _evict_style_model():
    if _STYLE_MODEL_CACHE["model"] is not None:
        _STYLE_MODEL_CACHE["model"] = None
        _STYLE_MODEL_CACHE["tokenizer"] = None
        _STYLE_MODEL_CACHE["key"] = None
        gc.collect()
        torch.cuda.empty_cache()


def get_style_model_cached(style, lang2, dir="/workspace/"):
    key = (style, lang2)
    if _STYLE_MODEL_CACHE["key"] == key and _STYLE_MODEL_CACHE["model"] is not None:
        print(f"  [cache] reusing resident classifier for {key}")
        return _STYLE_MODEL_CACHE["model"], _STYLE_MODEL_CACHE["tokenizer"]
    _evict_style_model()
    model, tokenizer = get_style_model(style, lang2, dir=dir)
    _STYLE_MODEL_CACHE.update(key=key, model=model, tokenizer=tokenizer)
    return model, tokenizer


def score_candidates(model, tokenizer, texts, batch_size=8):
    """Run inference and return a list of scalar scores (raw logits from score head)."""
    scores = []
    device = next(model.parameters()).device

    for i in tqdm(range(0, len(texts), batch_size), total=(len(texts) + batch_size - 1) // batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)

        logits = out.logits.float().cpu().view(-1)
        scores.extend(logits.tolist())

    return scores


def load_formal_data(dir="../"):
    languages = ["en", "fr", "it", "pt"]

    data = {lang: ([], []) for lang in languages}
    labels = {lang: [] for lang in languages}
    df_en = pd.read_csv(dir + "data/labeled_data/formal/en.csv")
    data["en"] = df_en["text"].tolist()
    labels["en"] = df_en["label"].tolist()

    for lang in languages[1:]:
        df = pd.read_csv(dir + f"/data/labeled_data/formal/{lang}.csv")
        data[lang] = df["text"].tolist()
        labels[lang] = df["label"].tolist()
    return data, labels


def is_multi_candidate(translations_for_key):
    if len(translations_for_key) == 0:
        return False
    return isinstance(translations_for_key[0], list)


_INTIMACY_LANGS = {"English", "Spanish", "Portuguese", "Italian", "French", "Chinese"}


def infer_style_and_split(pkl_path):
    fname = os.path.basename(pkl_path)

    m = re.search(r'_(test|train)(?:_\d+)?\.pkl$', fname)
    split = m.group(1) if m else "test"

    try:
        obj = pickle.load(open(pkl_path, "rb"))
    except Exception as e:
        return None, split, f"unreadable ({e})"

    if isinstance(obj, dict) and "translations" in obj and isinstance(obj["translations"], dict):
        inner = obj["translations"]
    elif isinstance(obj, dict):
        inner = obj
    else:
        return None, split, "not a dict"

    langs = set()
    for k in inner.keys():
        if "->" not in str(k):
            continue
        for part in str(k).split("->"):
            langs.add(part.strip())

    if not langs:
        return None, split, "no src->tgt keys"

    if langs & _INTIMACY_LANGS:
        style = "intimacy"
    elif langs & {"es", "ja", "zh"}:
        style = "politeness"
    elif langs & {"fr", "it", "pt"}:
        style = "formal"
    else:
        style = None  # only 'en' or unrecognised -> ambiguous, caller skips

    return style, split, f"langs={sorted(langs)}"


def main(args):
    def load_source_split():
        if args.source_split == "formal":
            return load_formal_data()
        if args.source_split == "train":
            return load_train_data(args.style)
        if args.source_split == "test":
            return load_test_data(args.style)
        if args.style == "formal":
            return load_formal_data()
        return load_test_data(args.style)

    # ---- load the raw pickle ----
    if args.pkl_path:
        loaded = pickle.load(open(args.pkl_path, "rb"))
    elif args.method == "direct" and "Llama" not in args.model:
        loaded = pickle.load(open(f"../data/{args.model}_{args.style}_translated_data.pkl", "rb"))
    else:
        loaded = pickle.load(open(f"translations/{args.method}_{args.model}_{args.style}.pkl", "rb"))

    # ---- detect format ----
    def is_v2(obj):
        if isinstance(obj, dict) and "translations" in obj and isinstance(obj["translations"], dict):
            inner = obj["translations"]
        elif isinstance(obj, dict):
            inner = obj
        else:
            return False, None
        for k, v in inner.items():
            if isinstance(v, list) and len(v) and isinstance(v[0], dict) \
               and "source_text" in v[0] and "candidates" in v[0]:
                return True, inner
            return False, None
        return False, None

    v2, v2_inner = is_v2(loaded)

    SOURCE_FROM_FILE = {}   # key -> (source_texts, source_labels) when v2

    if v2:
        meta = loaded.get("meta", {}) if isinstance(loaded, dict) and "meta" in loaded else {}
        print(f"[setup] Detected v2 source-carried translation file. meta={meta}")
        translations = {}
        for key, rows in v2_inner.items():
            translations[key] = [r["candidates"] if isinstance(r["candidates"], list)
                                 else [r["candidates"]] for r in rows]
            SOURCE_FROM_FILE[key] = (
                [r["source_text"] for r in rows],
                [float(r["source_label"]) for r in rows],
            )
        data_test, labels_test = None, None
    else:
        print("[setup] Detected v1 legacy bare-list translation file.")
        print(f"        -> re-loading source from split '{args.source_split}'.")
        translations = loaded
        if args.source_split == "auto":
            raise SystemExit(
                "ERROR: v1 file requires an explicit --source_split (test|train|formal). "
                "The translation pkl filename usually ends in _test or _train; use that.")
        data_test, labels_test = load_source_split()

    # Helper: get the (source_texts, source_labels) for a key, from whichever source.
    def source_for(key):
        if v2:
            return SOURCE_FROM_FILE[key]
        lang1 = key.split("->")[0]
        if lang1 not in data_test or lang1 not in labels_test:
            raise AlignmentError(
                f"[align] source lang {lang1!r} not found in loaded split "
                f"(have {list(data_test.keys())}). Wrong --source_split or style?")
        return data_test[lang1], labels_test[lang1]

    results = {}

    for key in list(translations.keys()):
        print(f"\n===== Evaluating {key} =====")

        lang1 = key.split("->")[0]
        lang2 = key.split("->")[1]

        raw = translations[key]
        multi = is_multi_candidate(raw)
        n_candidates = len(raw[0]) if multi else 1

        print(f"Multi-candidate mode: {multi}  |  candidates per item: {n_candidates}")
        print(f"Num items: {len(raw)}")

        # --- Verify alignment BEFORE scoring (cheap, fails fast) ---
        src_texts, src_labels = source_for(key)

        expected = None
        if not v2 and args.verify_split:
            try:
                other = ("train" if args.source_split == "test" else "test")
                other_loader = {"train": load_train_data, "test": load_test_data}[other]
                od, _ = other_loader(args.style)
                this_d, _ = source_for(key)  # texts only used
                k = min(20, len(this_d))
                this_match = sum(1 for i in range(k)
                                 if str(this_d[i]).strip() == str(src_texts[i]).strip())
                other_match = 0
                if lang1 in od:
                    kk = min(20, len(this_d), len(od[lang1]))
                    other_match = sum(1 for i in range(kk)
                                      if str(od[lang1][i]).strip() == str(src_texts[i]).strip())
                print(f"  [split-check] '{args.source_split}' self-match={this_match}/{k}, "
                      f"'{other}' match={other_match}/{k}")
                if other_match > this_match:
                    msg = (f"[split-check] The '{other}' split matches the stored source better "
                           f"than '{args.source_split}'. You likely picked the WRONG split. "
                           f"Re-run with --source_split {other}.")
                    if not args.allow_misaligned:
                        raise AlignmentError(msg)
                    print("  " + msg)
            except AlignmentError:
                raise
            except Exception as e:
                print(f"  [split-check] skipped ({e})")

        verify_alignment(
            key, raw, src_texts, src_labels,
            strict=not args.allow_misaligned,
            expected_source_texts=expected,
        )

        print(f"Loading style model for (style={args.style}, lang={lang2})...")
        model, tokenizer = get_style_model_cached(args.style, lang2)
        print("Model ready.")

        # Shape: (n_candidates, n_items)
        all_candidate_scores = []

        for cand_idx in range(n_candidates):
            print(f"\n--- Candidate {cand_idx} ---")

            if multi:
                cand_texts = [row[cand_idx] for row in raw]
            else:
                cand_texts = list(raw)

            error_count = sum(1 for t in cand_texts if "ERROR" in t)
            print(f"Error translations: {error_count}")

            cand_texts = [t.split("\n")[0] for t in cand_texts]

            print("Running inference...")
            pred_scores = score_candidates(model, tokenizer, cand_texts, batch_size=4)
            all_candidate_scores.append(pred_scores)

            print(f"Sample scores: {pred_scores[:3]}")

        # Transpose from (n_candidates, n_items) -> (n_items, n_candidates)
        per_item_scores = [list(s) for s in zip(*all_candidate_scores)]

        results[key] = per_item_scores

        print(f"\nSaved {len(per_item_scores)} items, each with {n_candidates} scores.")
        print(f"Sample: item 0 scores = {per_item_scores[0]}")

    training_data = {}

    for key in list(translations.keys()):
        lang1 = key.split("->")[0]

        raw = translations[key]
        per_item_scores = results[key]
        source_texts, source_labels = source_for(key)

        multi = is_multi_candidate(raw)

        raw_chk, source_texts, source_labels = verify_alignment(
            key, raw, source_texts, source_labels,
            strict=not args.allow_misaligned,
        )
        per_item_scores = per_item_scores[:len(source_texts)]

        items = []
        for i, (scores, label, src) in enumerate(zip(per_item_scores, source_labels, source_texts)):
            candidates = raw_chk[i] if multi else [raw_chk[i]]
            best_idx = int(np.argmax(scores))

            item = {
                "source_text": src,
                "source_label": float(label),
                "candidates": candidates,
                "style_scores": scores,
                "best_candidate_idx": best_idx,
                "score_deltas": [s - float(label) for s in scores],
            }
            items.append(item)

        training_data[key] = items
        print(f"\n{key}: built {len(items)} training examples")

        # Quick sanity correlation per key -- this IS the steerability number
        # (Pearson between requested source label and measured style score).
        if len(items) >= 3:
            lab = np.array([it["source_label"] for it in items])
            sc0 = np.array([it["style_scores"][0] for it in items])
            if sc0.std() > 1e-9 and lab.std() > 1e-9:
                r = pearsonr(lab, sc0)[0]
                print(f"  sanity Pearson(label, style_scores[0]) = {r:.4f}")
            else:
                print("  sanity Pearson: undefined (constant label or score)")
        print(f"  Sample item 0:")
        print(f"    source_label : {items[0]['source_label']}")
        print(f"    style_scores : {items[0]['style_scores']}")
        print(f"    best_idx     : {items[0]['best_candidate_idx']}")
        print(f"    score_deltas : {items[0]['score_deltas']}")

    # ------------------------------------------------------------------
    # Save both outputs (uniquely named per input pkl)
    # ------------------------------------------------------------------
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    tag = getattr(args, "out_tag", None) or f"{args.model}_{args.method}_{args.style}"

    scores_path = os.path.join(out_dir, f"{tag}_style_scores.pkl")
    pickle.dump(results, open(scores_path, "wb"))
    print(f"\nSaved per-item scores to {scores_path}")

    train_path = os.path.join(out_dir, f"{tag}_train_examples.pkl")
    pickle.dump(training_data, open(train_path, "wb"))
    print(f"Saved training examples to {train_path}")

    print("\n=== Summary ===")
    for key, items in training_data.items():
        scores_arr = np.array([it["style_scores"] for it in items])
        labels_arr = np.array([it["source_label"] for it in items])
        best_scores = scores_arr[np.arange(len(items)), [it["best_candidate_idx"] for it in items]]
        # Headline steerability number per direction.
        lab = labels_arr
        sc0 = scores_arr[:, 0]
        corr = (pearsonr(lab, sc0)[0]
                if (lab.std() > 1e-9 and sc0.std() > 1e-9) else float("nan"))
        print(f"  {key}:")
        print(f"    items          : {len(items)}")
        print(f"    Pearson r      : {corr:.4f}")
        print(f"    mean best score: {best_scores.mean():.4f}")
        print(f"    mean label     : {labels_arr.mean():.4f}")
        print(f"    mean delta     : {(best_scores - labels_arr).mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Hunyuan-MT-7B",
                        help="Only used for output naming when --out_tag is not derived from a file.")
    parser.add_argument("--style", type=str, default="all",
                        choices=["politeness", "intimacy", "formal", "all"],
                        help="In folder mode this is a FILTER: 'all' processes every style; "
                             "set it to process only one. Style is auto-detected per file regardless.")
    parser.add_argument("--method", type=str, default="rasta",
                        choices=["fewshot", "vanilla", "rasta", "direct", "all"])
    parser.add_argument("--pkl_path", type=str, default=None,
                        help="Evaluate a SINGLE pkl. If omitted, --translations_dir is scanned.")
    parser.add_argument("--translations_dir", type=str,
                        default="/gpfs/home2/tvarelanunes/llm-translate/MasterThesis/pipeline/translations",
                        help="Folder of translation pkls to evaluate (used when --pkl_path is not set).")
    parser.add_argument("--out_dir", type=str,
                        default="/gpfs/home2/tvarelanunes/llm-translate/MasterThesis/pipeline/data/test_data/grpo",
                        help="Where *_style_scores.pkl and *_train_examples.pkl are written.")
    parser.add_argument("--source_split", type=str, default="auto",
                        choices=["auto", "test", "train", "formal"],
                        help="Override per-file split detection. 'auto' = read _test/_train from the filename.")
    parser.add_argument("--allow_misaligned", action="store_true",
                        help="Do NOT hard-error on length mismatch; truncate to the shortest list.")
    parser.add_argument("--verify_split", action="store_true", default=True,
                        help="For v1 files, content-check the chosen split against the other. On by default.")
    parser.add_argument("--no_verify_split", dest="verify_split", action="store_false",
                        help="Disable the split content-check.")
    parser.add_argument("--create_tables", action="store_true")
    args = parser.parse_args()

    # Capture the user-supplied filter/override before per-file mutation.
    style_filter = args.style
    split_override = args.source_split

    def run_one(pkl_path):
        style, split, info = infer_style_and_split(pkl_path)
        print(f"\n################ {os.path.basename(pkl_path)}")
        print(f"  detected: style={style}, split={split}  ({info})")

        if style is None:
            print("  !! could not infer style from language keys; SKIPPING.")
            return (pkl_path, "SKIPPED (style?)")
        if style_filter != "all" and style_filter != style:
            print(f"  (filter) --style={style_filter}; skipping this {style} file.")
            return (pkl_path, f"SKIPPED (filtered: {style})")

        args.style = style
        args.source_split = split if split_override == "auto" else split_override
        args.pkl_path = pkl_path
        args.out_tag = os.path.splitext(os.path.basename(pkl_path))[0]

        try:
            main(args)
            return (pkl_path, "OK")
        except Exception as e:
            traceback.print_exc()
            return (pkl_path, f"FAILED ({e})")

    summary = []
    if args.pkl_path:
        summary.append(run_one(args.pkl_path))
    else:
        pkls = sorted(glob.glob(os.path.join(args.translations_dir, "*.pkl")))
        # Never re-ingest our own outputs if they happen to share the folder.
        pkls = [p for p in pkls
                if not p.endswith("_style_scores.pkl")
                and not p.endswith("_train_examples.pkl")]
        print(f"[batch] Found {len(pkls)} translation pkls in {args.translations_dir}")
        for p in pkls:
            summary.append(run_one(p))

    _evict_style_model()

    print("\n================ BATCH SUMMARY ================")
    for p, status in summary:
        print(f"  {status:24s} {os.path.basename(p)}")