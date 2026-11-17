"""3-line English caption (title / question? / save), 0 hashtags, from the episode JSON.
Uses the repo's gymform/caption.py build + lint (the same lint the publisher runs)."""
import importlib.util
import os
from .common import REPO

_spec = importlib.util.spec_from_file_location("gf_caption", os.path.join(REPO, "gymform", "caption.py"))
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)


def build(ep):
    return _m.build(ep["caption"])


def lint(text):
    return _m.lint(text)
