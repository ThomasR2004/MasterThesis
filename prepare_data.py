
import argparse
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm


BASE_DIR   = Path("/gpfs/home2/tvarelanunes/llm-translate/MasterThesis/pipeline/OPUS/Combined")
OUTPUT_DIR = Path("/gpfs/home2/tvarelanunes/llm-translate/MasterThesis/pipeline/processed")


PAIRS = ["es-ja", "en-fr", "it-pt", "en-es", "es-fr"]


SPLITS = ["train", "valid"]

LANG_NAMES = {
    "en": "English",
    "es": "Spanish",
    "ja": "Japanese",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
}


def make_user_content(src_lang: str, tgt_lang: str, src_text: str) -> str:
    src = LANG_NAMES[src_lang]
    tgt = LANG_NAMES[tgt_lang]
    return (
        f"Your task is to translate a given piece of text from {src} to {tgt}.\n"
        f"Keep in mind that style and tone vary across cultures, so a direct "
        f"word-for-word translation may not always preserve the intended meaning "
        f"and register of the original text.\n\n"
        f"This is the text you need to translate:\n"
        f"{src_text}\n\n"
        f"Now, translate the text into {tgt}. Output only the translation."
    )


def row_to_examples(row: pd.Series):
    examples = []
    lang1, lang2 = str(row["lang1"]), str(row["lang2"])
    s1, s2 = str(row["sentence1"]).strip(), str(row["sentence2"]).strip()
    if not s1 or not s2:
        return examples
    if lang1 not in LANG_NAMES or lang2 not in LANG_NAMES:
        return examples
    # Direction 1: lang1 -> lang2
    examples.append({
        "messages":   [{"role": "user", "content": make_user_content(lang1, lang2, s1)}],
        "completion": s2,   # no leading space - chat template handles the seam
        "src_lang":   lang1,
        "tgt_lang":   lang2,
    })
    # Direction 2: lang2 -> lang1
    examples.append({
        "messages":   [{"role": "user", "content": make_user_content(lang2, lang1, s2)}],
        "completion": s1,
        "src_lang":   lang2,
        "tgt_lang":   lang1,
    })
    return examples


def process_split(pair: str, split: str):
    parquet_path = BASE_DIR / pair / f"{split}.parquet"
    if not parquet_path.exists():
        # Skip-with-warning instead of crashing, so a missing split for one
        # pair doesn't abort the whole batch.
        print(f"  [skip] {pair}/{split}: parquet not found ({parquet_path})")
        return None
    print(f"Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    print(f"  Rows loaded: {len(df):,}")

    out_dir = OUTPUT_DIR / pair
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.jsonl"

    total = 0
    dropped_empty = 0
    dropped_unknown_lang = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{pair}/{split}"):
            s1 = str(row["sentence1"]).strip()
            s2 = str(row["sentence2"]).strip()
            lang1 = str(row["lang1"])
            lang2 = str(row["lang2"])
            if not s1 or not s2:
                dropped_empty += 1
                continue
            if lang1 not in LANG_NAMES or lang2 not in LANG_NAMES:
                dropped_unknown_lang += 1
                continue
            for ex in row_to_examples(row):
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                total += 1

    print(f"  Dropped (empty)        : {dropped_empty:,}")
    print(f"  Dropped (unknown lang) : {dropped_unknown_lang:,}")
    print(f"  Written examples       : {total:,} -> {out_path}\n")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIRS + ["all"], default="all",
                        help="Which language pair to process (default: all).")
    parser.add_argument("--split", choices=SPLITS + ["all"], default="all",
                        help="Which split to process (default: all).")
    args = parser.parse_args()

    pairs  = PAIRS  if args.pair  == "all" else [args.pair]
    splits = SPLITS if args.split == "all" else [args.split]

    for pair in pairs:
        for split in splits:
            process_split(pair, split)
    print("Done. Processed files are in:", OUTPUT_DIR)