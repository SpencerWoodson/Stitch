#!/usr/bin/env python3
import sys
import subprocess
import os

STITCH_PATH = os.path.expanduser("~/stitch")
VENV_PYTHON = os.path.join(STITCH_PATH, "llama-env", "bin", "python")
MAIN_SCRIPT = os.path.join(STITCH_PATH, "scripts", "main.py")

# Pass all CLI arguments to main.py
args = [VENV_PYTHON, MAIN_SCRIPT] + sys.argv[1:]
subprocess.run(args)
