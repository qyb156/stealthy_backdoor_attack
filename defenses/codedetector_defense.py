#!/usr/bin/env python3
"""
codedetector_defense.py — CodeDetector baseline defence (RQ4, Table 6).

CodeDetector (Li et al., 2022) uses integrated gradients over a trained victim model to
attribute importance to each input token; tokens with abnormally high attribution are
mined as potential triggers. Because the original CodeDetector code is not public
(see Section 6.4), we provide a faithful re-implementation on top of a Hugging Face
sequence-classification victim and Captum's IntegratedGradients.

For each input, attributions are aggregated per token; tokens whose attribution exceeds
mean + k*std are flagged. The script reports sample-level precision / recall against
ground-truth poison labels and, if --trigger-tokens is given, token-level precision /
recall against the true trigger tokens.

ASTT builds its trigger from structural AST nodes rather than salient lexical tokens, so
the highest-attribution tokens are ordinary keywords (e.g. `return`) and identifiers, and
the defence cannot isolate a trigger token sequence.

Input
-----
JSONL, one object per line: {"code": "<source>", "label": 0|1}

Usage
-----
    python codedetector_defense.py --input candidates.jsonl \
        --model_dir ../Defect-detection/code/saved_models --k 3.0
"""
import argparse
import json

import torch


def load_victim(model_dir):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, model.to(device), device


def token_attributions(tok, model, device, code, target=None):
    """Integrated-gradients attribution per token w.r.t. the predicted class."""
    from captum.attr import IntegratedGradients

    enc = tok(code, return_tensors="pt", truncation=True, max_length=400).to(device)
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask")
    emb_layer = model.get_input_embeddings()
    inputs_emb = emb_layer(input_ids)
    baseline_emb = torch.zeros_like(inputs_emb)

    def forward_emb(inputs_embeds):
        out = model(inputs_embeds=inputs_embeds, attention_mask=attn)
        return out.logits

    with torch.no_grad():
        pred = int(model(inputs_embeds=inputs_emb, attention_mask=attn).logits.argmax(-1))
    target = pred if target is None else target

    ig = IntegratedGradients(forward_emb)
    atts = ig.attribute(inputs_emb, baselines=baseline_emb, target=target, n_steps=50)
    scores = atts.sum(dim=-1).squeeze(0)                    # per-token attribution
    toks = tok.convert_ids_to_tokens(input_ids.squeeze(0))
    return toks, scores.detach().cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--model_dir", required=True, help="trained victim classifier (HF dir)")
    ap.add_argument("--k", type=float, default=3.0, help="flag tokens above mean + k*std")
    ap.add_argument("--trigger-tokens", nargs="*", default=None)
    args = ap.parse_args()

    tok, model, device = load_victim(args.model_dir)

    tp = fp = fn = tn = 0
    tok_tp = tok_fp = tok_fn = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            toks, scores = token_attributions(tok, model, device, obj["code"])
            thr = scores.mean() + args.k * scores.std()
            flagged = [t for t, s in zip(toks, scores) if s > thr]
            pred_poison = len(flagged) > 0
            is_poison = int(obj["label"]) == 1

            tp += pred_poison and is_poison
            fp += pred_poison and not is_poison
            fn += (not pred_poison) and is_poison
            tn += (not pred_poison) and not is_poison

            if args.trigger_tokens is not None:
                truth = set(args.trigger_tokens)
                flag = {t.lstrip("Ġ▁") for t in flagged}
                tok_tp += len(flag & truth)
                tok_fp += len(flag - truth)
                tok_fn += len(truth - flag)

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    print("CodeDetector defence (k=%.1f)" % args.k)
    print("  sample-level  precision=%.3f  recall=%.3f  (TP=%d FP=%d FN=%d TN=%d)"
          % (p, r, tp, fp, fn, tn))
    if args.trigger_tokens is not None:
        tp_p = tok_tp / (tok_tp + tok_fp) if (tok_tp + tok_fp) else 0.0
        tp_r = tok_tp / (tok_tp + tok_fn) if (tok_tp + tok_fn) else 0.0
        print("  token-level   precision=%.3f  recall=%.3f" % (tp_p, tp_r))


if __name__ == "__main__":
    main()
