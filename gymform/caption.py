"""Caption builder + lint for @gymform.3d (experience-question style).

Config "caption" may be a dict (preferred) or a legacy string:
  {"title": "one-line hook statement",
   "question": "ONE personal-experience question ending with ?",
   "save": "Save this for your next shoulder day."}
Format (Day 6+): EXACTLY 3 lines separated by blank lines: title / question? / save. No body.
Rules: 100% English, 0 hashtags, no DM-reward CTAs, no vote hooks (1/2, Comment "X").
"""
import re

VOTE = re.compile(r'\b(drop|comment|type|reply)\b[^.\n]{0,40}(\b[12]\b|"[A-Z]+"|\b[A-Z]{3,}\b)|\b1 or 2\b|\b1\s*\(|\bor 2\b', re.I)
DM = re.compile(r"\bDM\b|direct message|I'?ll send you|link in bio", re.I)


def build(cap):
    if isinstance(cap, str):
        return cap.strip() + "\n"
    if cap.get("body"):
        raise ValueError("caption body not allowed: 3-line format (title / question / save)")
    return "\n\n".join([cap["title"].strip(), cap["question"].strip(), cap["save"].strip()]) + "\n"


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
    blocks = text.strip().split("\n\n")
    if len(blocks) != 3 or any("\n" in b or not b.strip() for b in blocks):
        errs.append("must be exactly 3 lines separated by blank lines: title / question / save (no body text)")
    elif not blocks[1].strip().endswith("?"):
        errs.append("line 2 must be the experience question ending with ?")
    elif not blocks[2].strip().lower().startswith("save"):
        errs.append("line 3 must be the save line")
    return errs


# ---- spoken end question (Day 6+): the final TTS line asks the caption's experience question out loud ----
SPOKEN_DEFAULT = "Where do you feel it most?"
_PRON = {"them": "it", "they": "it", "these": "it", "those": "it"}


def spoken_question(cfg, spec=None):
    """Short (<= max_words) spoken version of the caption question. Explicit cfg["question_spoken"] wins;
    otherwise the shortest '?' clause of caption.question, pronouns normalised; fallback SPOKEN_DEFAULT."""
    rules = ((spec or {}).get("caption_rules") or {}).get("spoken_question", {})
    maxw = int(rules.get("max_words", 6))
    if cfg.get("question_spoken"):
        return cfg["question_spoken"].strip()
    cap = cfg.get("caption") or {}
    q = cap.get("question", "") if isinstance(cap, dict) else ""
    clauses = [c.strip(" ,.?") for c in re.split(r",|\band\b|\bor\b", q) if c.strip(" ,.?")]
    best = None
    for c in sorted(clauses, key=lambda c: len(c.split())):
        words = [_PRON.get(w.lower(), w) for w in c.split()]
        if words and words[0].lower() in ("what", "where", "when", "how", "which", "do", "does", "is", "can") and len(words) <= maxw:
            best = " ".join(words)
            break
    if not best:
        return rules.get("default", SPOKEN_DEFAULT)
    return best[0].upper() + best[1:] + "?"


def est_speech_s(text, sec_per_syl=0.16, pad=0.08):
    """Duration estimate for the locked voice (ChristopherNeural +24%, atempo 1.08): ~0.16 s/syllable (fit on Day 5)."""
    syl = sum(max(1, len(re.findall(r"[aeiouy]+", w.lower())) - (1 if w.lower().endswith("e") and len(w) > 3 else 0))
              for w in re.findall(r"[A-Za-z']+", text))
    num = len(re.findall(r"\d+", text)) * 2
    return (syl + num) * sec_per_syl + pad


def spoken_plan(cfg, spec):
    """Planned timing of the final step line + spoken question (estimates; tts.synth re-checks with real durations)."""
    r = spec["caption_rules"]["spoken_question"]
    q = spoken_question(cfg, spec)
    t4, l4 = [(float(t), s) for t, s in cfg["tts_lines"]][-1]
    end4 = t4 + est_speech_s(l4)
    sh = 5.0 * int(cfg.get("n_steps", 4)) - 20.0   # t568u: window moves with the reel length (2 steps = 10 s)
    qs = max(end4 + float(r["gap_after_step4_s"]), float(r.get("min_start_s", 0)) + sh)
    return {"question": q, "step4_end_est": round(end4, 2), "q_start_est": round(qs, 2),
            "q_end_est": round(qs + est_speech_s(q), 2), "latest_end_s": float(r["latest_end_s"]) + sh}


def lint_spoken(cfg, spec):
    """Errors for the spoken end question (empty list = OK / not required before from_day)."""
    r = spec["caption_rules"].get("spoken_question")
    if not r or int(cfg.get("day", 0)) < int(r.get("from_day", 6)):
        return []
    errs, p = [], spoken_plan(cfg, spec)
    q = p["question"]
    if not q.endswith("?") or q.count("?") != 1:
        errs.append("spoken question must be ONE question ending with ?")
    if len(q.split()) > int(r.get("max_words", 6)):
        errs.append(f"spoken question > {r['max_words']} words: {q!r}")
    if any(ord(ch) >= 0x1100 for ch in q) or "#" in q or DM.search(q) or VOTE.search(q):
        errs.append("spoken question must be plain English, no DM/vote hook")
    if p["q_end_est"] > p["latest_end_s"]:
        errs.append(f"spoken question would end ~{p['q_end_est']}s > {p['latest_end_s']}s: shorten the Step 4 line "
                    f"(ends ~{p['step4_end_est']}s) or the question")
    return errs
