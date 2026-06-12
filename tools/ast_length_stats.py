#!/usr/bin/env python3
"""
ast_length_stats.py — Reproduce the AST feature-vector length statistics of Section 6.1
and render the length-distribution figure referenced there.

For each input code snippet, the script:
  1. parses it into an AST (javalang for Java),
  2. keeps only the control-flow-related node types of Table 1 (in traversal order),
  3. records the length of that filtered node sequence (the "feature vector" length).

It then prints, for the chosen cut-off l, the share of snippets with length < l
(e.g. 84.2% < 3 for code translation, 79.8% < 4 for defect detection) and saves a
histogram + empirical CDF to a PNG.

Usage
-----
    # one Java method per line (e.g. CodeXGLUE train.java-cs.txt.java)
    python ast_length_stats.py --input ../code-to-code-trans/dataset/train.java --l 3

    # JSONL with a code field
    python ast_length_stats.py --input train.jsonl --field func --l 4 --out defect_len.png

The 84.2% / 79.8% coverage figures and the saved PNG are exactly the artefacts behind
Section 6.1 and its length-distribution figure.
"""
import argparse
import json
import os
import sys

# Table 1: control-flow-related AST node types and their integer encodings.
NODE_TABLE = {
    "DoStatement": 1,
    "ForControl": 2,
    "FormalParameter": 3,
    "ForStatement": 4,
    "IfStatement": 5,
    "ReturnStatement": 6,
    "SwitchStatement": 7,
    "SwitchStatementCase": 8,
    "WhileStatement": 9,
}


def parse_java(code):
    """Return a javalang AST for a Java snippet, wrapping it in a class if needed."""
    import javalang
    for src in (code, "public class _Wrap_ {\n%s\n}" % code):
        try:
            return javalang.parse.parse(src)
        except Exception:
            continue
    return None


def feature_sequence(tree):
    """Depth-first traversal keeping only Table-1 node types, in order."""
    seq = []
    for _, node in tree:
        name = type(node).__name__
        if name in NODE_TABLE:
            seq.append(NODE_TABLE[name])
    return seq


def iter_snippets(path, field):
    if path.endswith(".jsonl") or path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield obj[field]
    else:  # one snippet per line
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="corpus file (.jsonl or one-snippet-per-line)")
    ap.add_argument("--field", default="func", help="JSON field holding the code (jsonl input)")
    ap.add_argument("--l", type=int, default=3, help="length cut-off to report coverage for")
    ap.add_argument("--out", default="ast_length_distribution.png", help="output figure path")
    args = ap.parse_args()

    lengths, n_parse_fail = [], 0
    for code in iter_snippets(args.input, args.field):
        tree = parse_java(code)
        if tree is None:
            n_parse_fail += 1
            continue
        lengths.append(len(feature_sequence(tree)))

    if not lengths:
        sys.exit("No snippet could be parsed; check --input / --field.")

    n = len(lengths)
    import numpy as np
    arr = np.asarray(lengths)
    cov = (arr < args.l).mean() * 100.0
    print("parsed snippets : %d  (parse failures: %d)" % (n, n_parse_fail))
    print("mean length     : %.2f   median: %d   max: %d" % (arr.mean(), int(np.median(arr)), arr.max()))
    print("coverage(< l=%d): %.1f%%   -> %d of %d snippets are padded (not truncated)"
          % (args.l, cov, int((arr < args.l).sum()), n))
    print("truncated(>= l) : %.1f%%" % ((arr >= args.l).mean() * 100.0))

    # Histogram + empirical CDF, with the chosen cut-off marked.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(6, 4))
    bins = np.arange(0, arr.max() + 2) - 0.5
    ax1.hist(arr, bins=bins, color="#4C72B0", alpha=0.7, edgecolor="white")
    ax1.set_xlabel("AST feature-vector length")
    ax1.set_ylabel("count", color="#4C72B0")
    ax2 = ax1.twinx()
    xs = np.arange(0, arr.max() + 1)
    cdf = np.array([(arr <= x).mean() for x in xs])
    ax2.plot(xs, cdf, color="#C44E52", marker="o", ms=3)
    ax2.set_ylabel("empirical CDF", color="#C44E52")
    ax2.set_ylim(0, 1.02)
    ax1.axvline(args.l - 0.5, color="black", ls="--", lw=1)
    ax1.annotate("l = %d  (%.1f%% below)" % (args.l, cov),
                 xy=(args.l - 0.5, ax1.get_ylim()[1] * 0.9),
                 xytext=(args.l + 0.3, ax1.get_ylim()[1] * 0.9))
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print("figure written  : %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
