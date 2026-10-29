"""Caption builder + lint for @gymform.3d (experience-question style).

Config "caption" may be a dict (preferred) or a legacy string:
  {"title": "one-line hook statement",
   "body": ["Mistake 1: ...", "Mistake 2: ...", "The fix: ..."],
   "question": "ONE personal-experience question ending with ?",
   "save": "Save this for your next shoulder day."}
Rules: 100% English, 0 hashtags, no DM-reward CTAs, no vote hooks (1/2, Comment "X").
"""
import re

VOTE = re.compile(r'\b(drop|comment|type|reply)\b[^.\n]{0,40}(\b[12]\b|"[A-Z]+"|\b[A-Z]{3,}\b)|\b1 or 2\b|\b1\s*\(|\bor 2\b', re.I)
DM = re.compile(r"\bDM\b|direct message|I'?ll send you|link in bio", re.I)


def build(cap):
    if isinstance(cap, str):
        return cap.strip() + "\n"
    parts = [cap["title"].strip(), "\n".join(b.strip() for b in cap.get("body", [])), cap["question"].strip()]
    if cap.get("save"):
        parts.append(cap["save"].strip())
    return "\n\n".join(p for p in parts if p) + "\n"


def lint(text):
    errs = []
    lines = [l for l in text.strip().split("\n") if l.strip()]
    if "#" in text:
        errs.append("hashtag")
    if any(ord(ch) >= 0x1100 and not (0x2000 <= ord(ch) <= 0x2BFF) for ch in text):
        errs.append("non-English chars")
    if DM.search(text):
        errs.append("DM/reward CTA")
    if VOTE.search(text):
        errs.append("vote-style hook")
    if lines and "?" in lines[0]:
        errs.append("first line must be a one-line title, not a question")
    if lines and len(lines[0]) > 90:
        errs.append("title too long")
    if text.count("?") != 1:
        errs.append(f"need exactly 1 experience question (found {text.count('?')})")
    return errs
