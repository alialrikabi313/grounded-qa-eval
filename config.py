# -*- coding: utf-8 -*-
"""الإعدادات. عدّل هنا فقط عند تغيير العيّنة أو النموذج أو عتبات التقييم."""

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# مدخلات من arabic-rag-bench (ويكيبيديا العربية + أسئلة بأدلة حرفية)
BENCH_DATA = Path(os.environ.get("ARB_DATA_DIR", _HERE.parent / "arabic-rag-bench-data"))
CORPUS_JSONL = BENCH_DATA / "wiki_sample_300.jsonl"
QUESTIONS_JSONL = BENCH_DATA / "questions.jsonl"

# المخرجات خارج الريبو: العناصر تحوي نصوص ويكيبيديا
DATA_DIR = Path(os.environ.get("GQA_DATA_DIR", _HERE.parent / "grounded-qa-eval-data"))
ITEMS_JSONL = DATA_DIR / "items.jsonl"
RUNS_DIR = DATA_DIR / "runs"
MANIFEST_JSON = _HERE / "manifest.json"
RESULTS_MD = _HERE / "results.md"

# سؤال واحد لكل مقالة لتنويع المصادر؛ كل سؤال يعطي عنصرين: قابل للإجابة وغير قابل
N_QUESTIONS = 80
SEED = 0

EMBED_MODEL = "BAAI/bge-m3"
EMBED_DEVICE = os.environ.get("GQA_DEVICE")  # None = تلقائي
MAX_CHUNK_WORDS = 250
EVIDENCE_WINDOW_WORDS = 6

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = "qwen3.8:27b"
REQUEST_TIMEOUT_SEC = 900
# حدّ الإخراج: جواب وجملة دليل فقط؛ يمنع نموذجاً ثرثاراً من احتكار الوقت
MAX_OUTPUT_TOKENS = 400

# اقتباس النموذج يُعدّ «الدليل الصحيح» إن تجاوز F1 على الكلمات هذه العتبة مع اقتباس الذهب.
# 0.5 تسمح بإعادة صياغة طفيفة وتقصي النسخ العشوائي؛ مبدئية
QUOTE_F1_THRESHOLD = 0.5

BOOTSTRAP_ROUNDS = 2000
