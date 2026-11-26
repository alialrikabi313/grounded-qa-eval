# -*- coding: utf-8 -*-
"""اختبارات التقييم على حالات حقيقية: عناصر من ويكيبيديا العربية وردود فعلية من qwen3.8:27b
(tests/data). الحالة الأبرز: سؤال دخل الفرد في بلدة، فأجاب النموذج من مقطع بلدة أخرى."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scoring import newcombe_diff, score_item, summarize, wilson
from textutil import token_f1

_DATA = Path(__file__).resolve().parent / "data"


def _load(name: str) -> dict[str, dict]:
    with (_DATA / name).open(encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


ITEMS = _load("real_items.jsonl")
RUNS = _load("real_runs.jsonl")


def _score(item_id: str) -> dict:
    return score_item(ITEMS[item_id], RUNS[item_id]["parsed"])


def test_answer_from_wrong_towns_passage_is_failed_abstention_not_fabrication():
    # السؤال عن دخل الفرد في بلدة، والمقطع لبلدة أخرى: النموذج نسخ جملة موجودة فعلاً في المقطع
    # (اقتباس صحيح حرفياً) لكنه أجاب بلا سند للسؤال: فشل امتناع لا اختلاق اقتباس
    s = _score("2308528-neg")
    assert s["answered"] and not s["correct_abstention"]
    assert not s["quote_fabricated"]


def test_false_abstention_on_answerable_item_scores_zero():
    # الحالة نفسها بمقطعها الصحيح: امتنع النموذج وكان يجب أن يجيب
    s = _score("2308528-pos")
    assert s["abstained"] and not s["grounded_correct"]


def test_grounded_correct_answer_with_matching_evidence():
    assert _score("615029-pos")["grounded_correct"]


def test_correct_abstention_on_hard_negative():
    assert _score("615029-neg")["correct_abstention"]


def test_unparseable_reply_fails_everything_and_is_counted():
    item = ITEMS["615029-pos"]
    s = score_item(item, None)
    assert s["format_error"] and not s["grounded_correct"] and not s["answered"]


def test_quote_not_in_passage_is_flagged_fabricated():
    item = ITEMS["615029-pos"]
    fake = {"answerable": True, "answer": "x", "evidence_quote": "جملة لا وجود لها في المقطع إطلاقاً"}
    s = score_item(item, fake)
    assert s["quote_fabricated"] and not s["grounded_correct"]


def test_summary_on_the_four_real_items():
    items = list(ITEMS.values())
    r = summarize(items, RUNS)
    assert (r["n_pos"], r["n_neg"]) == (2, 2)
    # صحيح واحد من قابلَين، وامتناع صحيح واحد من غير قابلَين
    assert r["grounded_correct"][0] == 0.5 and r["abstention"][0] == 0.5
    assert r["balanced"][0] == 0.5


def test_wilson_matches_textbook_value():
    # القيمة المرجعية لـ 8/10 بفترة Wilson 95%: (0.490, 0.943)
    lo, hi = wilson(8, 10)
    assert round(lo, 3) == 0.490 and round(hi, 3) == 0.943


def test_f1_ignores_arabic_punctuation():
    assert token_f1("رفض الحوثيون، لاحقًا.", "رفض الحوثيون لاحقًا") == 1.0


def test_newcombe_identical_real_counts_gives_interval_symmetric_about_zero():
    # 66/80 = grounded-correct الفعلي لتشغيل qwen على 80 سؤالاً (results.md)
    d, (lo, hi) = newcombe_diff(66, 80, 66, 80)
    assert d == 0 and lo < 0 < hi and abs(lo + hi) < 1e-12


def test_newcombe_is_antisymmetric_when_sides_swap():
    # 66/80 grounded-correct مقابل 79/80 امتناع (كلاهما من نتائج qwen الفعلية)
    d1, (lo1, hi1) = newcombe_diff(66, 80, 79, 80)
    d2, (lo2, hi2) = newcombe_diff(79, 80, 66, 80)
    assert abs(d1 + d2) < 1e-12 and abs(lo1 + hi2) < 1e-12 and abs(hi1 + lo2) < 1e-12


def test_newcombe_clear_gap_excludes_zero_and_stays_inside_unit_range():
    # 0.825 مقابل 0.9875 على 80 لكل منهما: فرق واضح؛ وWald كان سيعطي حداً > 1 قرب السقف
    d, (lo, hi) = newcombe_diff(79, 80, 66, 80)
    assert lo > 0 and -1 <= lo <= hi <= 1 and abs(d - (79 / 80 - 66 / 80)) < 1e-12
