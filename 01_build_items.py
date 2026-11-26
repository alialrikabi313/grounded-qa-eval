# -*- coding: utf-8 -*-
"""المرحلة 1: بناء عناصر التقييم (قابلة للإجابة وغير قابلة).

يقرأ: wiki_sample_300.jsonl و questions.jsonl (من arabic-rag-bench).
ينتج: items.jsonl (النصوص، خارج الريبو) و manifest.json (المعرّفات والبذرة، داخل الريبو).
الخطوات: اختيار سؤال واحد لكل مقالة ← مقطع الدليل = الفقرة الحاوية للاقتباس (قابل للإجابة)
← أقرب مقطع من مقالة أخرى بالتشابه (غير قابل للإجابة: سلبي صعب).
السلبي الصعب هو ما يعيده الاسترجاع حين يخطئ، فيقيس الامتناع في ظرف حقيقي لا سهل.
"""

import json
import random
import sys

import numpy as np

from config import (
    CORPUS_JSONL,
    DATA_DIR,
    EMBED_DEVICE,
    EMBED_MODEL,
    ITEMS_JSONL,
    MANIFEST_JSON,
    N_QUESTIONS,
    QUESTIONS_JSONL,
    SEED,
)
from textutil import evidence_window, norm, paragraph_chunks


def main() -> None:
    for path in (CORPUS_JSONL, QUESTIONS_JSONL):
        if not path.exists():
            sys.exit(f"ملف مفقود: {path} (يُنتج بمراحل arabic-rag-bench)")

    with CORPUS_JSONL.open(encoding="utf-8") as f:
        articles = {a["pageid"]: a for a in map(json.loads, f)}
    with QUESTIONS_JSONL.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    # مقاطع كل المقالات (لاختيار السلبيات من خارج مقالة السؤال)
    chunk_texts: list[str] = []
    chunk_page: list[int] = []
    for pid, art in articles.items():
        for c in paragraph_chunks(art["text"]):
            chunk_texts.append(c)
            chunk_page.append(pid)
    page_arr = np.array(chunk_page)

    # سؤال واحد لكل مقالة، بترتيب عشوائي ثابت بالبذرة
    rng = random.Random(SEED)
    by_page: dict[int, list[dict]] = {}
    for q in questions:
        by_page.setdefault(q["pageid"], []).append(q)
    picked = [rng.choice(qs) for pid, qs in sorted(by_page.items())]
    rng.shuffle(picked)

    # استيراد كسول: المكتبة ثقيلة
    from sentence_transformers import SentenceTransformer

    print(f"◄ تضمين {len(chunk_texts)} مقطعاً …")
    model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    c_vecs = model.encode(chunk_texts, batch_size=32, normalize_embeddings=True)

    items = []
    for q in picked:
        if len(items) >= N_QUESTIONS * 2:
            break
        window = evidence_window(q["quote"])
        positives = [
            c for c, p in zip(chunk_texts, chunk_page) if p == q["pageid"] and window in norm(c)
        ]
        if not positives:
            continue
        q_vec = model.encode(q["question"], normalize_embeddings=True)
        scores = c_vecs @ q_vec
        scores[page_arr == q["pageid"]] = -1.0  # السلبي من مقالة أخرى فقط
        neg_i = int(np.argmax(scores))
        base = f"{q['pageid']}"
        items.append({
            "id": f"{base}-pos", "pageid": q["pageid"], "question": q["question"],
            "passage": positives[0], "answerable": True, "gold_quote": q["quote"],
        })
        items.append({
            "id": f"{base}-neg", "pageid": q["pageid"], "question": q["question"],
            "passage": chunk_texts[neg_i], "answerable": False, "gold_quote": "",
            "neg_pageid": chunk_page[neg_i], "neg_score": round(float(scores[neg_i]), 3),
        })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with ITEMS_JSONL.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    MANIFEST_JSON.write_text(
        json.dumps(
            {"seed": SEED, "n_questions": len(items) // 2, "ids": [it["id"] for it in items]},
            indent=1,
        ),
        encoding="utf-8",
    )
    n = len(items) // 2
    print(f"✓ {n} سؤالاً → {len(items)} عنصراً ({n} قابل، {n} غير قابل) ← {ITEMS_JSONL}")


if __name__ == "__main__":
    main()
