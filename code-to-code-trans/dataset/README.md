# Code translation dataset (CodeXGLUE Java↔C#)

Public CodeXGLUE `code-to-code-trans` data (Microsoft, MIT-licensed), one method per line:

| file | lines |
|---|---|
| `train.java-cs.txt.java` / `.cs` | 10,295 |
| `valid.java-cs.txt.java` / `.cs` | 499 |
| `test.java-cs.txt.java` / `.cs`  | 1,000 |

Source: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans/data

The trigger-mining script `../code/preprocess_atttack_strategy3_final_kmeans.py` reads these
files from `../dataset/` (run it from the `code/` directory). The AST feature-vector length
statistics of Section 6.1 (8,669 of 10,295 inputs = 84.2% have length ≤ 3) are reproduced by
`tools/ast_length_stats.py --input train.java-cs.txt.java --l 3`.
