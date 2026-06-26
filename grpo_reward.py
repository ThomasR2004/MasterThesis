import os

import numpy as np
import torch


_DATA_KEYS = {
    "intimacy": {"en": "English", "es": "Spanish", "pt": "Portuguese",
                 "it": "Italian", "fr": "French", "zh": "Chinese"},
}


def resolve_data_key(style, lang):
    """Map a lang code to the key the loaders use for this style (per embed_data)."""
    return _DATA_KEYS.get(style, {}).get(lang, lang)



_LANG_FULLNAME = {
    "en": "English", "es": "Spanish", "ja": "Japanese", "zh": "Chinese",
    "fr": "French", "it": "Italian", "pt": "Portuguese",
}


def resolve_adapter_path(reward_dir, target_lang, style):
    """Locate <reward_dir>/training/<lang>/<style>_lora, trying both the 2-letter code
    and full-name folder variants (e.g. 'fr' vs 'French'; 'zh' vs 'Chinese' vs
    'Simplified Chinese'). Returns the first variant that actually contains
    adapter_config.json, else raises with the list of paths tried."""
    candidates = [
        target_lang,                              
        resolve_data_key(style, target_lang),     
        _LANG_FULLNAME.get(target_lang, target_lang),  
    ]
    if target_lang == "zh":
        candidates.append("Simplified Chinese")
    seen, ordered = set(), []
    for c in candidates:                          
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    tried = []
    for c in ordered:
        path = os.path.join(reward_dir, "training", c, f"{style}_lora")
        tried.append(path)
        if os.path.isfile(os.path.join(path, "adapter_config.json")):
            return path
    raise FileNotFoundError(
        f"No '{style}_lora' adapter (adapter_config.json) found for target_lang "
        f"{target_lang!r} under {os.path.join(reward_dir, 'training')}. Tried: {tried}"
    )


def normalize_labels_01(labels, scale=4.0):
    """Map a label list to [0,1]. Pass-through if already in range, else / scale
    (the politeness prompt is 'out of 4', so scale defaults to 4)."""
    arr = np.asarray(labels, dtype=float)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if lo >= -1e-6 and hi <= 1.0 + 1e-6:
        return arr
    return np.clip(arr / float(scale), 0.0, 1.0)


def _sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logistic(x, y, iters=100, l2=1e-6):
    """1-D Platt scaling: fit P = sigmoid(A*x + B) by IRLS (Newton). No extra deps."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([x, np.ones_like(x)])      # columns: [A, B]
    w = np.zeros(2)
    for _ in range(iters):
        p = _sigmoid(X @ w)
        W = np.clip(p * (1.0 - p), 1e-9, None)
        g = X.T @ (p - y) + l2 * w
        H = X.T @ (X * W[:, None]) + l2 * np.eye(2)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w_new = w - step
        if np.max(np.abs(w_new - w)) < 1e-9:
            w = w_new
            break
        w = w_new
    return float(w[0]), float(w[1])


def _binary_auc(scores, labels):
    """AUC via the Mann-Whitney statistic (ties count 0.5). Diagnostic for binary heads."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    pos, neg = s[y >= 0.5], s[y < 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = np.sum(pos[:, None] > neg[None, :])
    eq = np.sum(pos[:, None] == neg[None, :])
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def combine_reward(calibrated_style, target, adequacy, adequacy_threshold, floor):
    """Pure reward rule (kept separate so it is unit-testable without any model)."""
    out = []
    for cs, t, ad in zip(calibrated_style, target, adequacy):
        if ad is None or ad < adequacy_threshold:
            out.append(float(floor))               # hard gate
        else:
            out.append(float(-abs(float(cs) - float(t))))
    return out


class StyleTargetReward:
    def __init__(self, style, target_lang, reward_dir="/workspace/",
                 adequacy_threshold=0.5, floor=-2.0, label_scale=4.0,
                 calib_samples=1000, batch_size=16, calib_method="auto"):
        self.style = style
        self.target_lang = target_lang
        self.adequacy_threshold = float(adequacy_threshold)
        self.floor = float(floor)
        self.label_scale = float(label_scale)
        self.batch_size = int(batch_size)
        self.calib_method = str(calib_method)      # "auto" | "linear" | "logistic"

        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                  BitsAndBytesConfig, AutoModel)
        from peft import PeftModel

        # ---- style head (frozen reward model) ----
        self.tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.tok.pad_token_id = self.tok.eos_token_id
        self.tok.padding_side = "right"
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForSequenceClassification.from_pretrained(
            "mistralai/Mistral-7B-v0.1", num_labels=1,
            pad_token_id=self.tok.pad_token_id,
            quantization_config=bnb, device_map="auto")
        adapter_path = resolve_adapter_path(reward_dir, target_lang, style)
        print(f"[reward] loading {style} head from: {adapter_path}")
        self.rm = PeftModel.from_pretrained(base, adapter_path).eval()

        # ---- BGE-M3 for cross-lingual adequacy ----
        self.bge_tok = AutoTokenizer.from_pretrained("BAAI/bge-m3", trust_remote_code=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.bge = AutoModel.from_pretrained("BAAI/bge-m3", trust_remote_code=True).to(dev).eval()

        # ---- calibration: reward logit -> [0,1], fit on GROUND-TRUTH labels ----
        self.cal = self._fit_calibration(calib_samples)

    # -- scoring helpers ----------------------------------------------------
    @torch.no_grad()
    def _score_style(self, texts):
        scores, dev = [], next(self.rm.parameters()).device
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[i:i + self.batch_size], return_tensors="pt",
                           padding=True, truncation=True, max_length=512)
            enc = {k: v.to(dev) for k, v in enc.items()}
            scores.extend(self.rm(**enc).logits.float().cpu().view(-1).tolist())
        return scores

    @torch.no_grad()
    def _embed(self, texts):
        embs, dev = [], next(self.bge.parameters()).device
        for i in range(0, len(texts), self.batch_size):
            enc = self.bge_tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                               return_tensors="pt", max_length=512).to(dev)
            embs.append(self.bge(**enc)["pooler_output"].float().cpu().numpy())
        return np.vstack(embs)

    def _fit_calibration(self, n):
        from load_data import load_train_data
        data, labels = load_train_data(self.style)
        key = resolve_data_key(self.style, self.target_lang)
        texts = [t.split("\n")[0] for t in data[key][:n]]
        labs = np.asarray(normalize_labels_01(labels[key][:n], self.label_scale),
                          dtype=float)
        rewards = np.asarray(self._score_style(texts), dtype=float)

        if rewards.std() < 1e-9 or labs.std() < 1e-9:
            print("[reward] WARNING: degenerate logits or labels; calibration disabled.")
            return ("linear", 1.0, 0.0)

        uniq = np.unique(np.round(labs, 6))
        is_binary = uniq.size == 2 and set(uniq.tolist()).issubset({0.0, 1.0})
        method = self.calib_method
        if method == "auto":
            method = "logistic" if is_binary else "linear"

        r = float(np.corrcoef(rewards, labs)[0, 1])    # point-biserial when binary
        if method == "logistic":
            y = (labs >= 0.5).astype(float)
            A, B = _fit_logistic(rewards, y)
            auc = _binary_auc(rewards, y)
            print(f"[reward] calibration ({self.style}, logistic): "
                  f"P = sigmoid({A:.4f}*logit + {B:.4f}) "
                  f"(n={len(texts)}, point-biserial r={r:.3f}, AUC={auc:.3f})")
            if not is_binary:
                print("[reward] NOTE: logistic calibration on non-binary labels; "
                      "thresholding at 0.5 for the positive class.")
            if auc == auc and auc < 0.7:               # not-NaN and weak separation
                print(f"[reward] WARNING: reward head separates the classes weakly "
                      f"(AUC={auc:.3f}); the style target signal will be noisy.")
            return ("logistic", float(A), float(B))

        a, b = np.polyfit(rewards, labs, 1)            # label ~= a*reward + b
        print(f"[reward] calibration ({self.style}, linear): "
              f"label = {a:.4f}*logit + {b:.4f} "
              f"(n={len(texts)}, corr(logit,label)={r:.3f})")
        if r < 0.5:
            print("[reward] WARNING: reward head correlates weakly with labels "
                  f"(corr={r:.3f}); the style target signal will be noisy.")
        return ("linear", float(a), float(b))

    def _calibrate(self, rewards):
        method = self.cal[0]
        if method == "logistic":
            _, A, B = self.cal
            return _sigmoid(A * np.asarray(rewards, dtype=float) + B).tolist()
        _, a, b = self.cal
        return [min(1.0, max(0.0, a * float(r) + b)) for r in rewards]

    def _adequacy(self, sources, candidates):
        embs = self._embed(list(sources) + list(candidates))
        n = len(sources)
        S, C = embs[:n], embs[n:]
        out = []
        for s, c in zip(S, C):
            denom = (np.linalg.norm(s) + 1e-9) * (np.linalg.norm(c) + 1e-9)
            out.append(float(np.dot(s, c) / denom))
        return out

    # -- TRL reward entry point --------------------------------------------
    def __call__(self, prompts, completions, target=None, source_text=None, **kwargs):
        texts = []
        for c in completions:
            t = c if isinstance(c, str) else c[-1]["content"]
            texts.append(t.split("\n")[0].strip())

        cal = self._calibrate(self._score_style(texts))
        adq = self._adequacy(source_text, texts)
        return combine_reward(cal, target, adq, self.adequacy_threshold, self.floor)