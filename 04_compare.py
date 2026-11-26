# -*- coding: utf-8 -*-
"""المرحلة 4: مقارنة نفس النموذج على مجموعتَي أسئلة من مولّدَين مختلفين (qwen و Gemini).

يقرأ: items*.jsonl و runs*/ (مخرج المراحل 1–2 لكل مجموعة). ينتج: results_compare.md.
الأسئلة مختلفة بين المجموعتين فلا اقتران بين العناصر؛ الفرق (gemini − qwen) بفترة Newcombe.
نضيف قراءة على المقالات المشتركة فقط: نفس النصوص بأسئلة من مولّدَين، لتخفيف أثر اختلاف المقالات.
الأرقام المطلقة لا تُقارَن بين مجموعتين مختلفتين الصعوبة؛ المقارنة المعنيّة هي الفرق وفترته.
"""

import argparse
import json
import sys

from config import DEFAULT_MODEL, RESULTS_MD, SET_PATHS
from scoring import newcombe_diff, summarize


def load_set(name: str, model_file: str) -> tuple[list[dict], dict[str, dict]]:
    items_path, runs_dir = SET_PATHS[name]
    run_path = runs_dir / model_file
    for p in (items_path, run_path):
        if not p.exists():
            sys.exit(f"ملف مفقود: {p}")
    with items_path.open(encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    with run_path.open(encoding="utf-8") as f:
        rows = {r["id"]: r for r in map(json.loads, f)}
    # الأزواج المكتملة فقط، كما في 03_score.py
    paired = {it["pageid"] for it in items if f"{it['pageid']}-pos" in rows and f"{it['pageid']}-neg" in rows}
    return [it for it in items if it["pageid"] in paired], rows


def counts(summary: dict) -> dict[str, tuple[int, int]]:
    """(نجاحات، مجموع) لكل نسبة قابلة للمقارنة بفرق Newcombe."""
    n_pos, n_neg = summary["n_pos"], summary["n_neg"]
    fq, fa = summary["fabricated_quotes"]
    return {
        "Grounded-correct (answerable)": (round(summary["grounded_correct"][0] * n_pos), n_pos),
        "False abstentions (answerable)": summary["false_abstentions"],
        "Abstains on unanswerable": (round(summary["abstention"][0] * n_neg), n_neg),
        "Fabricated quotes (of answers)": (fq, fa),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-file", default=DEFAULT_MODEL.replace(":", "_").replace("/", "_") + ".jsonl")
    args = ap.parse_args()

    data = {name: load_set(name, args.model_file) for name in ("qwen", "gemini")}
    shared = {it["pageid"] for it in data["qwen"][0]} & {it["pageid"] for it in data["gemini"][0]}

    def summary_of(name: str, only: set[int] | None = None) -> dict:
        items, rows = data[name]
        subset = [it for it in items if only is None or it["pageid"] in only]
        return summarize(subset, rows)

    full = {n: summary_of(n) for n in data}
    lines = [
        f"Model: `{args.model_file[:-6]}`. Questions per set: qwen {full['qwen']['n_pos']}, gemini {full['gemini']['n_pos']} "
        f"(articles in both sets: {len(shared)}).",
        "",
        "| Metric | qwen-generated | gemini-generated | Δ (gemini − qwen), 95% CI |",
        "|---|---|---|---|",
    ]
    cq, cg = counts(full["qwen"]), counts(full["gemini"])
    for metric in cq:
        (k1, n1), (k2, n2) = cq[metric], cg[metric]
        if n1 == 0 or n2 == 0:
            lines.append(f"| {metric} | {k1}/{n1} | {k2}/{n2} | n/a |")
            continue
        d, (lo, hi) = newcombe_diff(k2, n2, k1, n1)
        mark = "" if lo <= 0 <= hi else " *"
        lines.append(f"| {metric} | {k1}/{n1} = {k1 / n1:.3f} | {k2}/{n2} = {k2 / n2:.3f} | {d:+.3f} ({lo:+.3f} to {hi:+.3f}){mark} |")
    lines += [
        f"| Balanced score | {full['qwen']['balanced'][0]:.3f} | {full['gemini']['balanced'][0]:.3f} | (see CIs in results*.md) |",
        "",
        "`*` = interval excludes 0.",
    ]
    if shared:
        sq, sg = summary_of("qwen", shared), summary_of("gemini", shared)
        lines += [
            "",
            f"Shared articles only ({len(shared)}): same passages, questions from each generator.",
            "",
            "| Set | Questions | Grounded-correct | Abstains on unanswerable | Balanced |",
            "|---|---|---|---|---|",
            f"| qwen | {sq['n_pos']} | {sq['grounded_correct'][0]:.3f} | {sq['abstention'][0]:.3f} | {sq['balanced'][0]:.3f} |",
            f"| gemini | {sg['n_pos']} | {sg['grounded_correct'][0]:.3f} | {sg['abstention'][0]:.3f} | {sg['balanced'][0]:.3f} |",
        ]
    out = RESULTS_MD.with_name("results_compare.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
