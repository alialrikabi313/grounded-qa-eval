# grounded-qa-eval

[![tests](https://github.com/alialrikabi313/grounded-qa-eval/actions/workflows/test.yml/badge.svg)](https://github.com/alialrikabi313/grounded-qa-eval/actions/workflows/test.yml)

> **TL;DR** — Does an LLM answer *only* when the passage supports it, and quote its evidence *verbatim*? This harness scores that automatically (no LLM judge), including the failure that matters in RAG: answering from a **wrong-but-similar passage**. A local 27B model abstained on 96–99% of hard negatives and fabricated almost no quotes (1 in 174 answers) on **two** question sets from different generators — but its false-refusal rate on answerable questions was 11% on one set and 2% on the other, so that number is a property of the question set, not of the model.

## What it measures

Each question yields two items from the same article pair:

- **Answerable:** the paragraph that contains the evidence.
- **Unanswerable (hard negative):** the *most similar paragraph from a different article* — what a retriever returns when it misses. Easy negatives (random passages) would overstate abstention.

The model must reply with JSON: `answerable`, a short `answer`, and one `evidence_quote` copied verbatim. All scoring is mechanical:

| Metric | Definition |
|---|---|
| **Grounded-correct** (answerable items) | answered **and** quote found verbatim in the passage **and** quote matches the gold evidence (word-F1 ≥ 0.5) |
| **Abstention** (unanswerable items) | model said `answerable: false` |
| **Fabricated quotes** | answered, but quote is not in the passage |
| **Balanced score** | mean of grounded-correct and abstention, bootstrap 95% CI (Wilson CIs for the proportions) |

## Result

80 questions → 160 items, Arabic Wikipedia, `qwen3.8:27b` via Ollama (CPU), temperature 0:

| Model | Balanced (95% CI) | Grounded-correct | Lenient match | False abstentions | Abstains on unanswerable | Fabricated quotes | Format errors | Median s/item |
|---|---|---|---|---|---|---|---|---|
| qwen3.8:27b | **0.906** (0.86–0.94) | 0.825 (0.73–0.89) | 0.812 (0.71–0.88) | 9/80 | 0.988 (0.93–1.00) | 0/72 | 0 | 27.9 |

**Failure anatomy of the 14 answerable misses:** 9 false abstentions (11%) and 5 evidence mismatches. The model is *conservative*: near-perfect on not answering when it shouldn't, at the price of refusing when it could answer. The single abstention failure was a census question answered from a *different town's* passage (similarity 0.48, above the median negative similarity of 0.44).

**Sensitivity check.** I first suspected the F1 ≥ 0.5 rule penalised quoting the right sentence at a different length, so I added a lenient metric (quote contains the middle window of the gold evidence) *after* seeing results. It scored slightly *lower* (0.812), so the conclusion does not depend on the matching rule. The strict rule stays the primary metric.

## Robustness: questions from a second generator

The 80 questions above were written by the model family under test (qwen), which could flatter it. I rebuilt the items from questions written by a different model family (`gemini-3.5-flash-lite`, same 300 articles, same prompt and verbatim-quote filter, from [arabic-rag-bench](https://github.com/alialrikabi313/arabic-rag-bench)) and ran the same `qwen3.8:27b`, same prompt, temperature 0. N was fixed at 100 questions (200 items) *before* running, by time budget (see [decisions.md](decisions.md)). Per-set tables: `results.md`, `results_gemini.md`; comparison: [results_compare.md](results_compare.md) (`04_compare.py`).

| Metric | qwen-generated (80 q) | gemini-generated (100 q) | Δ (gemini − qwen), 95% Newcombe CI |
|---|---|---|---|
| Grounded-correct (answerable) | 66/80 = 0.825 | 97/100 = 0.970 | +0.145 (+0.058 to +0.245) * |
| False abstentions (answerable) | 9/80 = 0.113 | 2/100 = 0.020 | −0.092 (−0.181 to −0.020) * |
| Abstains on unanswerable | 79/80 = 0.988 | 96/100 = 0.960 | −0.028 (−0.087 to +0.033) |
| Fabricated quotes (of answers) | 0/72 | 1/102 | +0.010 (−0.041 to +0.053) |
| Balanced score (95% CI) | 0.906 (0.86–0.94) | 0.965 (0.94–0.99) | |

`*` = interval excludes 0. The questions differ between sets (only 26 articles are shared), so there is no item pairing and **absolute numbers are not comparable across sets**; only the differences and their intervals are informative. Restricted to the 26 shared articles (same passages, different questions) the model scored 0.731 vs 1.000 grounded-correct, which is descriptive only at n = 26.

What this says:

- **Self-evaluation bias is not supported as an explanation.** The model did *better* on the other generator's questions, not worse.
- **The "11% false refusals" headline did not hold.** It fell to 2% (CI excludes 0). It was a property of the qwen-written question set. I did not isolate why those questions were harder.
- **What held across both sets:** abstention on hard negatives stayed high (79/80 and 96/100; the difference is within noise), and quote fabrication was rare (1 in 174 answers overall). The one flagged quote is a near-copy with an added leading «و»; the strict verbatim rule still counts it.
- **A new caveat from reading the failures.** All 4 hard negatives the model answered on the Gemini set are questions that name no entity ("the German player", "the city", "the episode", "the star"), so a similar passage from another article can plausibly answer them. Those failures partly reflect under-specified questions, not only model behaviour, and the "unanswerable by construction" assumption is weaker for such questions. I read the cases by hand and did not build an automatic filter.
- The lenient evidence metric stayed below the strict one on both sets (lenient vs strict: 0.812 vs 0.825 on the qwen set, 0.830 vs 0.970 on the Gemini set), so the strict F1 rule is not what makes the numbers look good.

## Limitations (read these before citing the numbers)

- **One model.** A leaderboard needs more; adding a provider needs only an HTTP call in `02_run_model.py`. No cloud model has been run (no API key was available), and I ship no untested provider code.
- **Question generators.** Two were used (qwen, gemini; see Robustness). Both are LLMs reading the same text with the same prompt, so shared biases are not controlled, and the questions were not human-reviewed.
- **Ceiling effect.** 79/80 hard negatives were rejected, so the first negative set barely separates strong models; harder negatives (same entity, different fact) are the next step.
- **Arabic only**, English instructions, 80 + 100 questions, Wikipedia text. One evaluated model.
- **Different runs, same settings but not identical hardware settings:** both ran on CPU through Ollama; the second used 12 threads. At temperature 0 this should not change outputs materially, but I did not verify it.
- Negatives are assumed unanswerable because they come from another article; a different article could in rare cases contain the answer.

## Run

```bash
pip install -r requirements.txt
python 01_build_items.py                 # needs arabic-rag-bench data (ARB_DATA_DIR)
python 02_run_model.py --model qwen3.8:27b   # needs Ollama; resumable
python 03_score.py                       # writes results.md

# second item set (questions from questions_gemini.jsonl of arabic-rag-bench)
GQA_SET=gemini python 01_build_items.py
GQA_SET=gemini GQA_NUM_GPU=0 GQA_NUM_THREAD=12 python 02_run_model.py --limit 200
GQA_SET=gemini python 03_score.py        # writes results_gemini.md
python 04_compare.py                     # writes results_compare.md
pytest tests
```

Ollama did not use the GPU on my machine (22 s/item in a GPU trial), so runs are CPU-only; `GQA_NUM_GPU` / `GQA_NUM_THREAD` pass `num_gpu` / `num_thread` to Ollama. Item texts live outside the repo (`../grounded-qa-eval-data`). `manifest.json` records the seed and item ids. Test fixtures are four real items and real model replies, with Wikipedia attribution in `tests/data/ATTRIBUTION.md` (CC BY-SA 4.0).
