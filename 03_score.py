# -*- coding: utf-8 -*-
"""المرحلة 3: تقييم آلي لكل ملف في runs/.

يقرأ: items.jsonl و runs/*.jsonl. ينتج: results.md.
المقاييس (كلها قابلة للتحقق آلياً، بلا قاضٍ نموذجي):
- grounded-correct (على القابل للإجابة): أجاب + اقتباسه موجود حرفياً في المقطع + يطابق دليل الذهب (F1 ≥ عتبة).
- abstention (على غير القابل): امتنع. عكسه «إجابة بلا سند».
- quote-fabrication (على كل ما أجاب): اقتباس غير موجود في المقطع.
- الدرجة المتوازنة = متوسط grounded-correct و abstention. فترات الثقة: Wilson للنِّسَب، bootstrap للمتوسط.
"""

import json
import sys

from config import ITEMS_JSONL, RESULTS_MD, RUNS_DIR
from scoring import summarize


def fmt(v: tuple[float, tuple[float, float]]) -> str:
    return f"{v[0]:.3f} ({v[1][0]:.2f}–{v[1][1]:.2f})"


def main() -> None:
    if not ITEMS_JSONL.exists():
        sys.exit(f"لا توجد عناصر: {ITEMS_JSONL}")
    with ITEMS_JSONL.open(encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    runs = sorted(RUNS_DIR.glob("*.jsonl")) if RUNS_DIR.exists() else []
    if not runs:
        sys.exit(f"لا توجد تشغيلات في {RUNS_DIR}")

    lines = [
        "| Model | Items done | Balanced score (95% CI) | Grounded-correct on answerable | Abstains on unanswerable | Fabricated quotes | Format errors | Median s/item |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for path in runs:
        with path.open(encoding="utf-8") as f:
            rows = {r["id"]: r for r in map(json.loads, f)}
        # نقيّم الأزواج المكتملة فقط كي لا ينحاز المتوسط المتوازن لنصف ناقص
        done_pages = {it["pageid"] for it in items if it["id"] in rows}
        paired = {p for p in done_pages if f"{p}-pos" in rows and f"{p}-neg" in rows}
        subset = [it for it in items if it["pageid"] in paired]
        if not subset:
            continue
        r = summarize(subset, rows)
        fq, fa = r["fabricated_quotes"]
        lines.append(
            f"| {path.stem} | {len(subset)} | {fmt(r['balanced'])} | {fmt(r['grounded_correct'])} "
            f"| {fmt(r['abstention'])} | {fq}/{fa} | {r['format_errors']} | {r['median_seconds']} |"
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
