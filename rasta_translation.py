import argparse
import pandas as pd
from load_data import load_train_data, load_test_data, load_train_embeddings, load_test_embeddings
from embed_data import get_closest_embeddings
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os, re

# -- Language key definitions ---------------------------------------------------
# Full sets used when no model filter applies.

TRANS_KEYS_POLITENESS = [
    'en->es',
    'en->ja', 'en->zh', 'es->en', 'es->ja', 'es->zh',
    'ja->en', 'ja->es', 'ja->zh', 'zh->en', 'zh->es', 'zh->ja',
]
TRANS_KEYS_INTIMACY = [
    'English->Spanish', 'English->Portuguese', 'English->Italian', 'English->French', 'English->Chinese',
    'Spanish->English', 'Spanish->Portuguese', 'Spanish->Italian', 'Spanish->French', 'Spanish->Chinese',
    'Portuguese->English', 'Portuguese->Spanish', 'Portuguese->Italian', 'Portuguese->French', 'Portuguese->Chinese',
    'Italian->English', 'Italian->Spanish', 'Italian->Portuguese', 'Italian->French', 'Italian->Chinese',
    'French->English', 'French->Spanish', 'French->Portuguese', 'French->Italian', 'French->Chinese',
    'Chinese->English', 'Chinese->Spanish', 'Chinese->Portuguese', 'Chinese->Italian', 'Chinese->French',
]
TRANS_KEYS_FORMAL = [
    'en->fr', 'en->it', 'en->pt', 'fr->en', 'fr->it', 'fr->pt',
    'it->en', 'it->fr', 'it->pt', 'pt->en', 'pt->fr', 'pt->it',
]

LANG_KEYS_POLITENESS = ['en', 'es', 'ja', 'zh']
LANG_KEYS_INTIMACY   = ['English', 'Spanish', 'Portuguese', 'Italian', 'French', 'Chinese']
LANG_KEYS_FORMAL     = ['en', 'fr', 'it', 'pt']


# -- Per-model language restrictions -------------------------------------------
# Maps a substring of the model/checkpoint name to the languages it was
# fine-tuned on, per style. Add a new entry here whenever you train a new
# specialised model  no other code changes needed.
#
# Format:
#   "<model_name_substring>": {
#       "<style>": [lang, ...],   # only these langs are used for that style
#   }
#
# If a style is omitted the full default lang list is used.
# If the model name doesn't match any entry, defaults are used for all styles.

MODEL_LANG_OVERRIDES = {
    # ------------------------------------------------------------------
    # INTIMACY
    # ------------------------------------------------------------------
    "hunyuan-mt-lora-en-zh": {
        "politeness": ["en", "zh"],
        "intimacy":   ["English", "Chinese"],
    },
    "hunyuan-mt-lora-en-zh": {
        "intimacy": ["English", "Chinese"],
    },

    # ------------------------------------------------------------------
    # POLITENESS
    # ------------------------------------------------------------------
    "hunyuan-mt-lora-es-ja": {
        "politeness": ["es", "ja"],
    },
    
    # ------------------------------------------------------------------
    # FORMALITY
    # ------------------------------------------------------------------
    "hunyuan-mt-lora-en-fr": {
        "formal": ["en", "fr"],
    },
    "hunyuan-mt-lora-it-pt": {
        "formal": ["it", "pt"],
    },
}


MODEL_DIRECTION_OVERRIDES = {
    
    # ------------------------------------------------------------------
    # BASE
    # ------------------------------------------------------------------

    "hunyuan-mt-7b": {
        "politeness": [("en", "zh"), ("zh", "en"), ("es", "ja"), ("ja", "es")],
        "intimacy":   [("English", "Chinese"), ("Chinese", "English"),
                       ("Spanish", "French"),  ("French", "Spanish")],
        "formal":     [("en", "fr"), ("fr", "en"), ("it", "pt"), ("pt", "it")],
    },
    # ------------------------------------------------------------------
    # INTIMACY
    # ------------------------------------------------------------------

    # Spanish -> French
    "grpo_es_fr_intimacy": {
        "intimacy": [("Spanish", "French")],
    },

    # French -> Spanish
    "grpo_fr_es_intimacy": {
        "intimacy": [("French", "Spanish")],
    },

    # English -> Chinese
    "grpo_en_zh_intimacy": {
        "intimacy": [("English", "Chinese")],
    },

    # Chinese -> English
    "grpo_zh_en_intimacy": {
        "intimacy": [("Chinese", "English")],
    },

    # ------------------------------------------------------------------
    # POLITENESS
    # ------------------------------------------------------------------

    # Spanish -> Japanese
    "grpo_es_ja_politeness": {
        "politeness": [("es", "ja")],
    },

    # Japanese -> Spanish
    "grpo_ja_es_politeness": {
        "politeness": [("ja", "es")],
    },

    # English -> Chinese
    "grpo_en_zh_politeness": {
        "politeness": [("en", "zh")],
    },

    # Chinese -> English
    "grpo_zh_en_politeness": {
        "politeness": [("zh", "en")],
    },

    # ------------------------------------------------------------------
    # FORMALITY
    # ------------------------------------------------------------------

    # English -> French
    "grpo_en_fr_formal": {
        "formal": [("en", "fr")],
    },

    # French -> English
    "grpo_fr_en_formal": {
        "formal": [("fr", "en")],
    },

    # Italian -> Portuguese
    "grpo_it_pt_formal": {
        "formal": [("it", "pt")],
    },

    # Portuguese -> Italian
    "grpo_pt_it_formal": {
        "formal": [("pt", "it")],
    },
}

# Number of translation candidates to generate per sentence.
# Increase here if you need more options to select from later.
N_TRANSLATIONS = 1


# -- Resolve language keys for a given model + style ---------------------------

def resolve_lang_keys(model_path: str, style: str) -> list[str]:
    model_tag = model_path.rstrip("/").split("/")[-1].lower()

    for pattern, style_map in MODEL_LANG_OVERRIDES.items():
        if pattern.lower() in model_tag:
            if style in style_map:
                langs = style_map[style]
                print(f"[lang filter] Model '{model_tag}' matched pattern '{pattern}' "
                      f"→ restricting {style} to {langs}")
                return langs
            # model matched but this style not in its map → not supported, skip it
            print(f"[lang filter] Model '{model_tag}' matched pattern '{pattern}' "
                  f"but '{style}' is not supported skipping.")
            return []

    defaults = {
        "politeness": LANG_KEYS_POLITENESS,
        "intimacy":   LANG_KEYS_INTIMACY,
        "formal":     LANG_KEYS_FORMAL,
    }
    print(f"[lang filter] No override for '{model_tag}' / '{style}' → using full default list")
    return defaults[style]


# -- Resolve allowed directions for a given model + style ----------------------

def resolve_directions(model_path: str, style: str) -> list[tuple[str, str]] | None:
    """
    Return the explicit src->tgt pairs allowed for this model+style, or None
    if no restriction applies (in which case all ordered pairs are used).
    Case-insensitive substring match on the last component of the model path.
    """
    model_tag = model_path.rstrip("/").split("/")[-1].lower()

    for pattern, style_map in MODEL_DIRECTION_OVERRIDES.items():
        if pattern.lower() in model_tag and style in style_map:
            dirs = style_map[style]
            print(f"[dir filter] Model '{model_tag}' matched pattern '{pattern}' "
                  f"→ restricting {style} to directions {dirs}")
            return dirs

    return None


# -- vLLM helpers ---------------------------------------------------------------

def build_model(model_name: str, lora_path: str | None = None) -> tuple[LLM, SamplingParams]:
    """
    Initialise the vLLM engine.
    If lora_path is provided the LoRA adapter is loaded on top of the base model.
    """
    kwargs = dict(
        max_num_seqs=8,
        max_model_len=1024,
        enforce_eager=True,
        tensor_parallel_size=1,   # split model across 2 GPUs
    )

    if lora_path:
        from vllm.lora.request import LoRARequest
        kwargs["enable_lora"] = True
        print(f"LoRA adapter will be loaded from: {lora_path}")

    model = LLM(model_name, **kwargs)

    # SamplingParams: n=N_TRANSLATIONS generates N candidates in one call.
    params = SamplingParams(
        n=N_TRANSLATIONS,
        temperature=0.6,
        top_p=0.9,
        max_tokens=300,
    )
    return model, params


def query_vllm(
    prompts: list[str],
    model: LLM,
    params: SamplingParams,
    lora_path: str | None = None,
) -> list[list[str]]:
    """
    Returns a list of lists: outer index = sentence, inner = N_TRANSLATIONS candidates.
    """
    chat_input = [[{"role": "user", "content": p}] for p in prompts]

    extra = {}
    if lora_path:
        from vllm.lora.request import LoRARequest
        extra["lora_request"] = LoRARequest("finetuned", 1, lora_path)

    outputs = model.chat(chat_input, sampling_params=params, **extra)

    # Each output has N_TRANSLATIONS candidates in .outputs
    return [[o.text for o in out.outputs] for out in outputs]


# -- Embedding alignment --------------------------------------------------------

def align_embedding(style, embedding, label, source_key, target_key, embeddings, labels):
    source_embeddings = embeddings[source_key]
    target_embeddings = embeddings[target_key]
    source_labels     = labels[source_key]
    target_labels     = labels[target_key]

    source_diffs = [abs(sl - label) for sl in source_labels]
    target_diffs = [abs(tl - label) for tl in target_labels]

    num_embs_source = int(len(source_embeddings) * 0.1)
    num_embs_target = int(len(target_embeddings) * 0.1)

    if style == "formality":
        num_embs_source = int(np.count_nonzero(np.array(source_diffs) == 0))
        num_embs_target = int(np.count_nonzero(np.array(target_diffs) == 0))

    source_indexes = np.argsort(source_diffs)[:num_embs_source]
    target_indexes = np.argsort(target_diffs)[:num_embs_target]

    source_mean = np.mean(source_embeddings[source_indexes], axis=0)
    target_mean = np.mean(target_embeddings[target_indexes], axis=0)

    return embedding + (target_mean - source_mean)


# -- Prompt loading -------------------------------------------------------------

_KEY_MAP = {
    "en": "English", "es": "Spanish", "ja": "Japanese",
    "zh": "Simplified Chinese", "fr": "French",  "it": "Italian", "pt": "Portuguese",
}

def load_prompt(style, source_key, target_key, text, label, rag_examples):
    template_map = {
        "politeness": "prompts/politeness_translation.txt",
        "intimacy":   "prompts/intimacy_translation.txt",
        "formal":     "prompts/formality_translation.txt",
    }
    if style in ("politeness", "formal"):
        source_key = _KEY_MAP.get(source_key, source_key)
        target_key = _KEY_MAP.get(target_key, target_key)

    with open(template_map[style]) as fh:
        prompt = fh.read()

    return prompt.format(
        source_key, target_key, text, label,
        source_key, target_key,
        rag_examples[0], rag_examples[1], rag_examples[2], rag_examples[3], rag_examples[4],
        label, target_key,
    )


# -- Leakage-safe RAG retrieval ------------------------------------------------

def get_rag_examples(
    style: str,
    target_key: str,
    data: dict,
    embeddings: dict,
    labels: dict,
    aligned_embedding: np.ndarray,
    label: float,
    exclude_index: int | None = None,
) -> list:
    tgt_data       = data[target_key]
    tgt_embeddings = embeddings[target_key]
    tgt_labels     = labels[target_key]

    if exclude_index is not None:
        # Temporarily remove the current sentence from the pool
        tgt_data       = [s for j, s in enumerate(tgt_data)       if j != exclude_index]
        tgt_embeddings = np.delete(tgt_embeddings, exclude_index, axis=0)
        tgt_labels     = [l for j, l in enumerate(tgt_labels)     if j != exclude_index]

    rag_examples, _, _ = get_closest_embeddings(
        style,
        tgt_data,
        tgt_embeddings,
        tgt_labels,
        aligned_embedding,
        label,
        5,
    )
    return rag_examples


# -- Core translation loop ------------------------------------------------------

def run_translation(
    style: str,
    lang_keys: list[str],
    model: LLM,
    params: SamplingParams,
    split: str = "test",
    lora_path: str | None = None,
    max_samples: int | None = None,
    directions: list[tuple[str, str]] | None = None,
) -> dict:
    train_data,      train_labels      = load_train_data(style)
    train_embeddings                   = load_train_embeddings(style)
    test_data,       test_labels       = load_test_data(style)
    test_embeddings                    = load_test_embeddings(style)

    if split == "train":
        # Sentences to translate = train set; RAG pool = train set (with self excluded)
        src_data       = train_data
        src_labels     = train_labels
        src_embeddings = train_embeddings
        rag_data       = train_data
        rag_embeddings = train_embeddings
        rag_labels     = train_labels
        print(f"  [split=train] Self-exclusion enabled current sentence will be "
              f"removed from the RAG pool at each step.")
    else:
        # Sentences to translate = test set; RAG pool = train set (no overlap)
        src_data       = test_data
        src_labels     = test_labels
        src_embeddings = test_embeddings
        rag_data       = train_data
        rag_embeddings = train_embeddings
        rag_labels     = train_labels

    results = {}

    for source_key in lang_keys:
        for target_key in lang_keys:
            if target_key == source_key:
                continue
            if directions is not None and (source_key, target_key) not in directions:
                continue

            trans_key = f"{source_key}->{target_key}"
            n_sentences = len(src_data[source_key])
            if max_samples is not None:
                n_sentences = min(n_sentences, max_samples)

            print(f"[{style}] Translating {trans_key}  "
                  f"(split={split}, {n_sentences} sentences, {N_TRANSLATIONS} candidates/sentence)")

            prompts = []
            for i in tqdm(range(n_sentences)):
                utterance = src_data[source_key][i]
                label     = src_labels[source_key][i]
                embedding = src_embeddings[source_key][i]

                aligned = align_embedding(
                    style, embedding, label,
                    source_key, target_key,
                    train_embeddings, train_labels,
                )

                # exclude index i only when the RAG pool == the source split
                exclude = i if split == "train" else None

                rag_examples = get_rag_examples(
                    style, target_key,
                    rag_data, rag_embeddings, rag_labels,
                    aligned, label,
                    exclude_index=exclude,
                )
                prompts.append(
                    load_prompt(style, source_key, target_key, utterance, round(label, 3), rag_examples)
                )

            # Each entry: list of N_TRANSLATIONS strings
            candidates = query_vllm(prompts, model, params, lora_path=lora_path)
            print(f"  ✓ {len(candidates)} sentences × {N_TRANSLATIONS} candidates")
            results[trans_key] = candidates

    return results


# -- Entry point ----------------------------------------------------------------

STYLE_CONFIG = {
    "politeness": LANG_KEYS_POLITENESS,
    "intimacy":   LANG_KEYS_INTIMACY,
    "formal":     LANG_KEYS_FORMAL,
}


def derive_model_tag(path: str) -> str:
    parts = path.rstrip("/").split("/")
    last = parts[-1]
    m = re.fullmatch(r"checkpoint-(\d+)", last)
    if m and len(parts) >= 2:
        return f"{parts[-2]}-c{m.group(1)}"
    return last


def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{root}_{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1
        
def main():
    parser = argparse.ArgumentParser(description="RASTA style-aware translation via vLLM")
    parser.add_argument("--model",  default="tencent/Hunyuan-MT-7B",
                        help="HuggingFace model id or path to base model")
    parser.add_argument("--lora",   default=None,
                        help="Path to LoRA adapter checkpoint (optional)")
    parser.add_argument("--style",  choices=list(STYLE_CONFIG.keys()), required=True,
                        help="Which style dimension to translate")
    parser.add_argument("--split",  choices=["test", "train"], default="test",
                        help="Which split to translate. "
                             "'train' enables self-exclusion from the RAG pool to prevent leakage.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of sentences to translate per language pair. "
                             "Useful for quick runs or debugging. Default: all sentences.")
    args = parser.parse_args()

    # Resolve which languages to use (auto-detects from model name)
    lang_keys = resolve_lang_keys(args.lora or args.model, args.style)

    # Resolve direction restriction (None = all ordered pairs)
    directions = resolve_directions(args.lora or args.model, args.style)

    print(f"Loading model : {args.model}")
    if args.lora:
        print(f"LoRA adapter  : {args.lora}")
    print(f"Style         : {args.style}")
    print(f"Split         : {args.split}")
    print(f"Languages     : {lang_keys}")
    if directions is not None:
        print(f"Directions    : {directions}")
    print(f"Candidates/sentence: {N_TRANSLATIONS}")

    if args.max_samples:
        print(f"Max samples/pair : {args.max_samples}")

    model, params = build_model(args.model, lora_path=args.lora)
    results = run_translation(args.style, lang_keys, model, params,
                              split=args.split, lora_path=args.lora,
                              max_samples=args.max_samples,
                              directions=directions)
    model_tag = derive_model_tag(args.lora or args.model)

    # Language pairs actually run, in the order they appear in the results
    # (so this stays correct under both --lang-key filtering and direction overrides).
    pair_tag = "_".join(k.replace("->", "-") for k in results.keys()) if results else "nopairs"

    out_path = (
        f"translations/rasta_{model_tag}_{args.style}_{pair_tag}_{args.split}.pkl"
    )
    out_path = unique_path(out_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fh:
        pd.to_pickle(results, fh)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()