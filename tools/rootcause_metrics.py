"""Thin wrapper (kept for the commands in docs/ROOTCAUSE_DAY6.md): the code lives in gymform/temporal_qc.py."""
import os, runpy, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
runpy.run_module("gymform.temporal_qc", run_name="__main__")
