#!/usr/bin/env python3
"""
compiler_defense.py — COMPILER baseline defence (RQ4, Table 6).

The COMPILER defence flags a sample as poisoned if it fails to parse/compile, on the
assumption that an injected trigger breaks syntactic validity. ASTT injects only
*syntactically valid* code into the OUTPUT and leaves the INPUT unchanged, so the
defence has nothing uncompilable to catch.

Given a labelled set of candidate samples, this script predicts "poison" for every
sample that fails to parse and reports precision / recall against the ground truth.

Input
-----
A JSONL file, one object per line:
    {"code": "<source code>", "label": 0|1}     # label 1 = actually poisoned

Usage
-----
    python compiler_defense.py --input candidates.jsonl --lang java
"""
import argparse
import json


def parses_java(code):
    import javalang
    for src in (code, "public class _W_ {\n%s\n}" % code):
        try:
            javalang.parse.parse(src)
            return True
        except Exception:
            continue
    return False


def parses_c(code):
    from tree_sitter import Language, Parser
    # Expects a prebuilt grammar at build/my-languages.so (see defenses/README.md).
    parser = Parser()
    parser.set_language(Language("build/my-languages.so", "c"))
    tree = parser.parse(bytes(code, "utf8"))
    return not tree.root_node.has_error


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="JSONL with fields code,label")
    ap.add_argument("--lang", choices=["java", "c"], default="java")
    args = ap.parse_args()

    compiles = parses_java if args.lang == "java" else parses_c

    tp = fp = fn = tn = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pred_poison = not compiles(obj["code"])   # uncompilable => predicted poison
            is_poison = int(obj["label"]) == 1
            if pred_poison and is_poison:
                tp += 1
            elif pred_poison and not is_poison:
                fp += 1
            elif (not pred_poison) and is_poison:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    print("COMPILER defence")
    print("  TP=%d FP=%d FN=%d TN=%d" % (tp, fp, fn, tn))
    print("  precision=%.3f  recall=%.3f" % (precision, recall))
    print("  (ASTT keeps all inputs syntactically valid -> precision=recall=0 expected.)")


if __name__ == "__main__":
    main()
