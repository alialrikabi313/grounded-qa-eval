# -*- coding: utf-8 -*-
"""الإعدادات. عدّل هنا فقط عند تغيير العيّنة أو النموذج أو عتبات التقييم."""

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# مدخلات من arabic-rag-bench (ويكيبيديا العربية + أسئلة بأدلة حرفية)
BENCH_DATA = Path(os.environ.get("ARB_DATA_DIR", _HERE.parent / "arabic-rag-bench-data"))
CORPUS_JSONL = BENCH_DATA / "wiki_sample_300.jsonl"

# مجموعتا عناصر من مولّدَي أسئلة مختلفين لقياس التحيّز الذاتي (النموذج المُقيَّم من عائلة qwen).
# الاسم الافتراضي qwen يُبقي أسماء الملفات الأصلية كما هي؛ غيرها يأخذ لاحقة (items_gemini.jsonl …)
ITEM_SET = os.environ.get("GQA_SET", "qwen")
_QUESTION_FILES = {"qwen": "questions.jsonl", "gemini": "questions_gemini.jsonl"}
if ITEM_SET not in _QUESTION_FILES:
    raise SystemExit(f"GQA_SET غير معروف: {ITEM_SET} (المتاح: {', '.join(_QUESTION_FILES)})")
QUESTIONS_JSONL = BENCH_DATA / _QUESTION_FILES[ITEM_SET]
_SUFFIX = "" if ITEM_SET == "qwen" else f"_{ITEM_SET}"

# المخرجات خارج الريبو: العناصر تحوي نصوص ويكيبيديا
DATA_DIR = Path(os.environ.get("GQA_DATA_DIR", _HERE.parent / "grounded-qa-eval-data"))
ITEMS_JSONL = DATA_DIR / f"items{_SUFFIX}.jsonl"
RUNS_DIR = DATA_DIR / f"runs{_SUFFIX}"
# مسارات كل المجموعات معاً: 04_compare.py يقرؤها كلها بصرف النظر عن GQA_SET
SET_PATHS = {
    name: (DATA_DIR / f"items{'' if name == 'qwen' else '_' + name}.jsonl",
           DATA_DIR / f"runs{'' if name == 'qwen' else '_' + name}")
    for name in _QUESTION_FILES
}
MANIFEST_JSON = _HERE / f"manifest{_SUFFIX}.json"
RESULTS_MD = _HERE / f"results{_SUFFIX}.md"

# سؤال واحد لكل مقالة لتنويع المصادر؛ كل سؤال يعطي عنصرين: قابل للإجابة وغير قابل.
# gemini: كل المقالات الصالحة (حدّها الأعلى 266) ثم يُحدَّد N المقيَّم مسبقاً بميزانية الوقت لا بالنتائج
N_QUESTIONS = 80 if ITEM_SET == "qwen" else 1000
SEED = 0

EMBED_MODEL = "BAAI/bge-m3"
EMBED_DEVICE = os.environ.get("GQA_DEVICE")  # None = تلقائي
MAX_CHUNK_WORDS = 250
EVIDENCE_WINDOW_WORDS = 6

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
# عدد طبقات الـGPU لـOllama: None = قرار Ollama التلقائي، 0 = معالج فقط (يترك الـGPU لغيره)
OLLAMA_NUM_GPU = int(os.environ["GQA_NUM_GPU"]) if os.environ.get("GQA_NUM_GPU") else None
DEFAULT_MODEL = "qwen3.8:27b"
REQUEST_TIMEOUT_SEC = 900
# حدّ الإخراج: جواب وجملة دليل فقط؛ يمنع نموذجاً ثرثاراً من احتكار الوقت
MAX_OUTPUT_TOKENS = 400

# اقتباس النموذج يُعدّ «الدليل الصحيح» إن تجاوز F1 على الكلمات هذه العتبة مع اقتباس الذهب.
# 0.5 تسمح بإعادة صياغة طفيفة وتقصي النسخ العشوائي؛ مبدئية
QUOTE_F1_THRESHOLD = 0.5

BOOTSTRAP_ROUNDS = 2000
