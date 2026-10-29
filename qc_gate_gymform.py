#!/usr/bin/env python3
"""Back-compat shim: the QC gate lives in gymform/qc.py (python3 -m gymform.qc <mp4> --qc-dir <dir>)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gymform.qc import main

if __name__ == "__main__":
    main()
