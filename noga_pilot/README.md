# NOGA 2025 Laya pilot

`run.py` runs the fixed, zero-shot multilingual Laya pilot against a CSV sample and a NOGA catalogue. It makes three section comparisons (English, option-order rotations, German), chooses the section arm with the strongest agreement to Cortex (ties go to the rotation arm), then follows parent links through division, group, class and type. A low confidence does not stop the path; an invalid child/error does. Missing purposes stay in the sample and are not classified.

The runner stores full choice options, probability vectors, choices and timings in DuckDB, and writes an Excel review sheet plus a summary sheet. The source CSV is read only. Code values are written as text. Known DE/FR/IT source language metadata is retained; rows marked `unknown` are checked with Laya's dependency-free local heuristic, and uncertain text remains `unknown`. Language is reported but never used for routing. Top-two paths are evaluated separately when the section or division probability margin is below 0.15; they do not alter the greedy result.

Example from an isolated Python 3.12 environment (install this checkout and `duckdb`, `openpyxl`, plus the Apple Silicon PyTorch build first):

```sh
env -i HOME=/tmp/laya-home PATH=/tmp/laya-venv/bin:/usr/bin:/bin \
  HF_HOME=/tmp/laya-hf USE_TF=0 python noga_pilot/run.py \
  --sample /path/to/sample.csv \
  --catalogue /path/to/noga_2025.csv \
  --xlsx /path/to/review.xlsx \
  --duckdb /path/to/results.duckdb \
  --revision <hugging-face-commit-sha>
```

The runner pins the Hugging Face revision and the multilingual subfolder through Laya's `Router`; use an isolated `HF_HOME` and an environment without yardr session variables. It selects MPS when PyTorch reports it available, otherwise CPU. Keep one run/model at a time on shared hardware.
