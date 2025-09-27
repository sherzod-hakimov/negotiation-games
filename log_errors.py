#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, Any, Iterable

LANG_DIRS = ["results_en", "results_de", "results_it"]
GAME_NAME = "hot_air_balloon"
TARGET_TYPES = {"missing tag", "rule violation"}  # normalized, case-insensitive

def find_repo_root(start: Path) -> Path:
    """Walk up until we see any results_* directory; else use CWD."""
    p = start.resolve()
    while True:
        if any((p / d).exists() for d in LANG_DIRS):
            return p
        if p.parent == p:
            return start.resolve()
        p = p.parent

def iter_interaction_files(results_dir: Path) -> Iterable[Path]:
    yield from results_dir.glob("**/interactions.json")

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def flatten_turns(turns):
    for round_msgs in turns or []:
        if isinstance(round_msgs, list):
            for msg in round_msgs:
                yield msg

def norm_type(s: Optional[str]) -> str:
    """Normalize action.type for matching (lowercase, underscores -> spaces, strip)."""
    if not s:
        return ""
    return s.replace("_", " ").strip().lower()

def main():
    repo_root = find_repo_root(Path(os.getcwd()))
    # model_folder -> {"missing tag": int, "rule violation": int}
    counts_by_model = defaultdict(lambda: {t: 0 for t in TARGET_TYPES})

    scanned = 0
    for lang_dir in LANG_DIRS:
        base = repo_root / lang_dir
        if not base.exists():
            continue
        for path in iter_interaction_files(base):
            data = load_json(path)
            if not data:
                continue

            meta = data.get("meta", {}) or {}
            if (meta.get("game_name") or "").strip() != GAME_NAME:
                continue

            model_folder = meta.get("results_folder") or "unknown_model"
            scanned += 1

            for msg in flatten_turns(data.get("turns")):
                if msg.get("from") == "GM" and msg.get("to") == "GM":
                    a_type = norm_type((msg.get("action") or {}).get("type"))
                    if a_type in TARGET_TYPES:
                        # store under the canonical target key (proper casing)
                        key = "missing tag" if a_type == "missing tag" else "rule violation"
                        counts_by_model[model_folder][key] += 1

    if scanned == 0:
        print("(no hot_air_balloon interactions.json files found under results_en/de/it)")
        return

    if not counts_by_model:
        print("(no GM→GM 'missing tag' or 'rule violation' events found)")
        return

    # Pretty print table
    models = sorted(counts_by_model.keys())
    header = ["Model (results_folder)", "missing tag", "rule violation", "total"]
    col_widths = [max(len(header[0]), max(len(m) for m in models))]
    # numeric columns fixed width
    num_w = 14
    print(f"{header[0]:<{col_widths[0]}}  {header[1]:>{num_w}}  {header[2]:>{num_w}}  {header[3]:>{num_w}}")
    print("-" * (col_widths[0] + 2 + num_w*3 + 2))

    for m in models:
        mt = counts_by_model[m]["missing tag"]
        rv = counts_by_model[m]["rule violation"]
        total = mt + rv
        print(f"{m:<{col_widths[0]}}  {mt:>{num_w}d}  {rv:>{num_w}d}  {total:>{num_w}d}")

    # Also print a grand total line
    g_mt = sum(counts_by_model[m]["missing tag"] for m in models)
    g_rv = sum(counts_by_model[m]["rule violation"] for m in models)
    g_total = g_mt + g_rv
    print("-" * (col_widths[0] + 2 + num_w*3 + 2))
    print(f"{'TOTAL':<{col_widths[0]}}  {g_mt:>{num_w}d}  {g_rv:>{num_w}d}  {g_total:>{num_w}d}")

if __name__ == "__main__":
    main()