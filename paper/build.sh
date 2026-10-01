#!/bin/bash
# Regenerate numbers, tables and figures from results/, then build the PDF.
# Usage (inside WSL, from the repo root):  bash paper/build.sh
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$HOME/ova/venv/bin/python}"
cd "$REPO"
"$PY" analysis/make_paper_numbers.py
"$PY" analysis/make_figures.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error paper.tex > build.log 2>&1 || { tail -30 build.log; exit 1; }
bibtex paper >> build.log 2>&1 || { tail -20 build.log; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >> build.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >> build.log 2>&1
grep -E "Warning: (Reference|Citation)|Overfull \\\\hbox \([0-9]{2,}" build.log | sort | uniq -c | tail -15 || true
grep -E "Output written" build.log | tail -1
