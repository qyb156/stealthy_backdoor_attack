#!/usr/bin/env python3
r"""
onion_defense.py — ONION baseline defence (RQ4, Table 6).

ONION (Qi et al., 2021) detects backdoor triggers as *outlier tokens*: for each token w_i
in an input, it measures the perplexity drop when w_i is removed (leave-one-out),
    f_i = PPL(sentence) - PPL(sentence \ w_i),
using a pretrained language model. A large positive f_i means removing w_i makes the text
much more natural, i.e. w_i is an unnatural inserted token (a likely trigger). Tokens with
f_i above a threshold are flagged.

Against a *token-level* trigger (dead code, renamed identifier) this fires, because the
inserted tokens are rare and inflate perplexity. Against ASTT it does not: ASTT inserts no
input tokens, so there is no unnatural token for ONION to surface.

This script computes, per sample, the maximum suspicion score; a sample is flagged if any
token exceeds --threshold. It reports precision / recall against ground-truth poison labels,
and (if --trigger-tokens is given) token-level precision / recall against the true trigger
tokens.

Input
-----
JSONL, one object per line: {"code": "<source>", "label": 0|1}

Usage
-----
    python onion_defense.py --input candidates.jsonl \
        --lm microsoft/CodeGPT-small-java --threshold 0.0
"""
import argparse
import json

import torch


class PerplexityScorer:
    def __init__(self, lm_name, device=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(lm_name)
        self.model = AutoModelForCausalLM.from_pretrained(lm_name).to(self.device).eval()

    @torch.no_grad()
    def ppl(self, text):
        ids = self.tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(self.device)
        if ids.size(1) < 2:
            return 0.0
        out = self.model(ids, labels=ids)
        return float(torch.exp(out.loss))


def suspicion_scores(scorer, code):
    """Leave-one-out perplexity reduction f_i for each whitespace token."""
    tokens = code.split()
    base = scorer.ppl(code)
    scores = []
    for i in range(len(tokens)):
        reduced = " ".join(tokens[:i] + tokens[i + 1:])
        scores.append(base - scorer.ppl(reduced))   # f_i: positive => suspicious
    return tokens, scores


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--lm", default="microsoft/CodeGPT-small-java")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="flag a sample if max token suspicion f_i exceeds this")
    ap.add_argument("--trigger-tokens", nargs="*", default=None,
                    help="ground-truth trigger tokens for token-level P/R (e.g. ret_var_)")
    args = ap.parse_args()

    scorer = PerplexityScorer(args.lm)

    tp = fp = fn = tn = 0
    tok_tp = tok_fp = tok_fn = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tokens, scores = suspicion_scores(scorer, obj["code"])
            flagged = [t for t, s in zip(tokens, scores) if s > args.threshold]
            pred_poison = len(flagged) > 0
            is_poison = int(obj["label"]) == 1

            tp += pred_poison and is_poison
            fp += pred_poison and not is_poison
            fn += (not pred_poison) and is_poison
            tn += (not pred_poison) and not is_poison

            if args.trigger_tokens is not None:
                truth = set(args.trigger_tokens)
                flag = set(flagged)
                tok_tp += len(flag & truth)
                tok_fp += len(flag - truth)
                tok_fn += len(truth - flag)

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    print("ONION defence (LM=%s, threshold=%.3f)" % (args.lm, args.threshold))
    print("  sample-level  precision=%.3f  recall=%.3f  (TP=%d FP=%d FN=%d TN=%d)"
          % (p, r, tp, fp, fn, tn))
    if args.trigger_tokens is not None:
        tp_p = tok_tp / (tok_tp + tok_fp) if (tok_tp + tok_fp) else 0.0
        tp_r = tok_tp / (tok_tp + tok_fn) if (tok_tp + tok_fn) else 0.0
        print("  token-level   precision=%.3f  recall=%.3f" % (tp_p, tp_r))


if __name__ == "__main__":
    main()
