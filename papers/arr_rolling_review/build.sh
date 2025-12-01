#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PDF=main.pdf

if command -v tectonic >/dev/null 2>&1; then
  echo "[build] using tectonic"
  tectonic -X compile --keep-intermediates --keep-logs --synctex main.tex
elif command -v pdflatex >/dev/null 2>&1; then
  echo "[build] using pdflatex (2x)"
  pdflatex -interaction=nonstopmode main.tex >/dev/null || true
  bibtex main || true
  pdflatex -interaction=nonstopmode main.tex >/dev/null || true
  pdflatex -interaction=nonstopmode main.tex >/dev/null || true
else
  echo "No LaTeX engine found. On Overleaf, just upload this folder. Locally, install tectonic or TeX Live." >&2
  exit 2
fi

ls -lh "$PDF" 2>/dev/null || true
