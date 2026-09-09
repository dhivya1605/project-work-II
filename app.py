"""
app.py
------
Streamlit Web Application & Research Dashboard Launcher.

Dispatches to model/app.py.
"""

import sys
import os

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "model"))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from app import main

if __name__ == "__main__":
    main()
