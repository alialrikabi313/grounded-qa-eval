Model: `qwen3.8_27b`. Questions per set: qwen 80, gemini 100 (articles in both sets: 26).

| Metric | qwen-generated | gemini-generated | Δ (gemini − qwen), 95% CI |
|---|---|---|---|
| Grounded-correct (answerable) | 66/80 = 0.825 | 97/100 = 0.970 | +0.145 (+0.058 to +0.245) * |
| False abstentions (answerable) | 9/80 = 0.113 | 2/100 = 0.020 | -0.092 (-0.181 to -0.020) * |
| Abstains on unanswerable | 79/80 = 0.988 | 96/100 = 0.960 | -0.028 (-0.087 to +0.033) |
| Fabricated quotes (of answers) | 0/72 = 0.000 | 1/102 = 0.010 | +0.010 (-0.041 to +0.053) |
| Balanced score | 0.906 | 0.965 | (see CIs in results*.md) |

`*` = interval excludes 0.

Shared articles only (26): same passages, questions from each generator.

| Set | Questions | Grounded-correct | Abstains on unanswerable | Balanced |
|---|---|---|---|---|
| qwen | 26 | 0.731 | 1.000 | 0.865 |
| gemini | 26 | 1.000 | 0.962 | 0.981 |
