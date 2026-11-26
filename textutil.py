# -*- coding: utf-8 -*-
"""دوال نصية مشتركة بين المراحل: تطبيع، تقطيع بالفقرات، نافذة الدليل، F1 على الكلمات."""

import re

from config import EVIDENCE_WINDOW_WORDS, MAX_CHUNK_WORDS

# عناوين ويكيبيديا النصّية: == عنوان ==
_HEADING_RE = re.compile(r"^=+\s*.+?\s*=+$")
# علامات الترقيم العربية واللاتينية؛ تُحذف عند مقارنة الكلمات فقط
_PUNCT_RE = re.compile(r"[\.,;:!?؟،؛\"'()\[\]«»…\-–—]")


def norm(s: str) -> str:
    return " ".join(s.split())


def paragraph_chunks(text: str) -> list[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip() and not _HEADING_RE.match(p.strip())]
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for p in paras:
        w = len(p.split())
        if cur and cur_words + w > MAX_CHUNK_WORDS:
            chunks.append(" ".join(cur))
            cur, cur_words = [], 0
        cur.append(p)
        cur_words += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def evidence_window(quote: str) -> str:
    # الوسط لا الطرفان: حدّ المقطع قد يقصّ طرف الاقتباس
    words = quote.split()
    mid = len(words) // 2
    half = EVIDENCE_WINDOW_WORDS // 2
    return " ".join(words[max(0, mid - half) : mid + half])


def _tokens(s: str) -> list[str]:
    return _PUNCT_RE.sub(" ", s).split()


def token_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 0.0
    common = 0
    g_left = list(g)
    for tok in p:
        if tok in g_left:
            g_left.remove(tok)
            common += 1
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)
