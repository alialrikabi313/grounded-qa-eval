# -*- coding: utf-8 -*-
"""المرحلة 2: تشغيل نموذج على العناصر.

يقرأ: items.jsonl. ينتج: runs/<model>.jsonl — سطر لكل عنصر (الرد الخام والحقول المحلَّلة).
الخطوات: طلب JSON من النموذج (جواب + اقتباس دليل) ← تحليل الرد ← حفظ فوراً.
قابل للاستئناف بمعرّف العنصر. فشل التحليل يُسجَّل خطأ صيغة ويُحتسب في التقييم، لا يُخفى.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

from config import (
    DEFAULT_MODEL,
    ITEMS_JSONL,
    MAX_OUTPUT_TOKENS,
    OLLAMA_NUM_GPU,
    OLLAMA_NUM_THREAD,
    OLLAMA_URL,
    REQUEST_TIMEOUT_SEC,
    RUNS_DIR,
)

# التعليمات بالإنجليزية لكل العناصر كي لا تتغير صياغة المهمة بتغير لغة المقطع
PROMPT = """You are given a passage and a question. Decide whether the passage itself contains the answer.

Reply with JSON only, in exactly this shape:
{{"answerable": true or false, "answer": "<short answer in the language of the question, or empty>", "evidence_quote": "<one sentence copied verbatim from the passage that supports the answer, or empty>"}}

Rules:
- Use only the passage. Do not use outside knowledge.
- If the passage does not contain the answer, set "answerable" to false and leave "answer" and "evidence_quote" empty.
- "evidence_quote" must be copied character for character from the passage.

Passage:
{passage}

Question: {question}"""


def ask_ollama(model: str, passage: str, question: str) -> str:
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": MAX_OUTPUT_TOKENS},
        # يُفرَّغ النموذج من الذاكرة بعد دقيقة خمول كي لا يحجز VRAM لغيره بعد انتهاء التشغيل
        "keep_alive": "60s",
        "messages": [
            {"role": "user", "content": PROMPT.format(passage=passage, question=question)}
        ],
    }
    if OLLAMA_NUM_GPU is not None:
        body["options"]["num_gpu"] = OLLAMA_NUM_GPU
    if OLLAMA_NUM_THREAD is not None:
        body["options"]["num_thread"] = OLLAMA_NUM_THREAD
    req = urllib.request.Request(
        OLLAMA_URL, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            return json.load(resp)["message"]["content"]
    except urllib.error.URLError as e:
        sys.exit(f"تعذّر الاتصال بـ Ollama على {OLLAMA_URL}: {e.reason}")


def parse_reply(raw: str) -> dict | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    # الصيغة المطلوبة بالضبط: حقل answerable منطقي، والباقي نصوص
    if not isinstance(obj, dict) or not isinstance(obj.get("answerable"), bool):
        return None
    return {
        "answerable": obj["answerable"],
        "answer": str(obj.get("answer") or ""),
        "evidence_quote": str(obj.get("evidence_quote") or ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="أول N عنصراً فقط (0 = الكل)")
    args = ap.parse_args()

    if not ITEMS_JSONL.exists():
        sys.exit(f"لا توجد عناصر: شغّل 01_build_items.py أولاً ({ITEMS_JSONL})")
    with ITEMS_JSONL.open(encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    if args.limit:
        items = items[: args.limit]

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / (args.model.replace(":", "_").replace("/", "_") + ".jsonl")
    done: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
    print(f"◄ {args.model}: {len(done)}/{len(items)} منجَز مسبقاً")

    with out_path.open("a", encoding="utf-8") as out:
        for i, it in enumerate(items, 1):
            if it["id"] in done:
                continue
            t0 = time.time()
            raw = ask_ollama(args.model, it["passage"], it["question"])
            parsed = parse_reply(raw)
            row = {"id": it["id"], "raw": raw, "parsed": parsed, "seconds": round(time.time() - t0, 1)}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"… {i}/{len(items)} ({row['seconds']}s)" + ("" if parsed else "  ✗ صيغة"), flush=True)

    print(f"✓ {out_path}")


if __name__ == "__main__":
    main()
