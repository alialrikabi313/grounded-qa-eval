# -*- coding: utf-8 -*-
"""دوال التقييم النقية: فترات Wilson وتصحيح العنصر وتلخيص التشغيل (مستقلة عن الملفات)."""

import math
import random

from config import BOOTSTRAP_ROUNDS, QUOTE_F1_THRESHOLD, SEED
from textutil import evidence_window, norm, token_f1


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe_diff(k1: int, n1: int, k2: int, n2: int, z: float = 1.96) -> tuple[float, tuple[float, float]]:
    """فرق نسبتين مستقلتين p1 − p2 بفترة Newcombe المركّبة من فترتَي Wilson (طريقة 10 في ورقته).

    نستعملها لأن المجموعتين أسئلة مختلفة (لا اقتران بين العناصر)، ولأن Wilson يبقى سليماً قرب 0 و1
    حيث ينهار تقريب Wald، وهو حال الامتناع هنا (≈ 0.99)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return d, (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def score_item(item: dict, parsed: dict | None) -> dict:
    """نتيجة عنصر واحد كقيم منطقية؛ ردّ بلا صيغة صحيحة = فشل في كل المقاييس."""
    abstained = parsed is not None and not parsed["answerable"]
    answered = parsed is not None and parsed["answerable"]
    quote = norm(parsed["evidence_quote"]) if parsed else ""
    quote_valid = bool(quote) and quote in norm(item["passage"])
    out = {"format_error": parsed is None, "answered": answered, "abstained": abstained,
           "quote_fabricated": answered and not quote_valid}
    if item["answerable"]:
        hit = answered and quote_valid and token_f1(quote, item["gold_quote"]) >= QUOTE_F1_THRESHOLD
        out["grounded_correct"] = hit
        # تحليل حساسية معلَن (أُضيف بعد رؤية النتائج، فليس المقياس الرئيسي): اقتباس يحوي نافذة الدليل
        # الوسطى يُقبل ولو اختلف طول الجملة؛ العتبة الصارمة F1 تظلم اقتباس نفس الجملة بطول آخر
        out["grounded_correct_lenient"] = answered and quote_valid and evidence_window(item["gold_quote"]) in quote
    else:
        out["correct_abstention"] = abstained
    return out


def summarize(items: list[dict], rows: dict[str, dict]) -> dict:
    scored = [(it, score_item(it, rows[it["id"]]["parsed"])) for it in items if it["id"] in rows]
    pos = [s for it, s in scored if it["answerable"]]
    neg = [s for it, s in scored if not it["answerable"]]
    answered = [s for _, s in scored if s["answered"]]
    k_pos = sum(s["grounded_correct"] for s in pos)
    k_len = sum(s["grounded_correct_lenient"] for s in pos)
    false_abstain = sum(s["abstained"] for s in pos)
    k_neg = sum(s["correct_abstention"] for s in neg)

    # bootstrap للمتوسط المتوازن بإعادة سحب كل مجموعة على حدة
    rng = random.Random(SEED)
    boots = []
    for _ in range(BOOTSTRAP_ROUNDS):
        a = sum(rng.choice(pos)["grounded_correct"] for _ in pos) / len(pos)
        b = sum(rng.choice(neg)["correct_abstention"] for _ in neg) / len(neg)
        boots.append((a + b) / 2)
    boots.sort()
    return {
        "n_pos": len(pos), "n_neg": len(neg),
        "grounded_correct": (k_pos / len(pos), wilson(k_pos, len(pos))),
        "grounded_lenient": (k_len / len(pos), wilson(k_len, len(pos))),
        "false_abstentions": (false_abstain, len(pos)),
        "abstention": (k_neg / len(neg), wilson(k_neg, len(neg))),
        "balanced": ((k_pos / len(pos) + k_neg / len(neg)) / 2,
                     (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))])),
        "fabricated_quotes": (sum(s["quote_fabricated"] for s in answered), len(answered)),
        "format_errors": sum(s["format_error"] for _, s in scored),
        "median_seconds": sorted(rows[it["id"]]["seconds"] for it, _ in scored)[len(scored) // 2],
    }
