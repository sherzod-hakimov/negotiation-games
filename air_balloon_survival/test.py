#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, Any, Iterable

# -------- model pretty names (as provided) --------
MODEL_NAME_MAP = {
    # GPT-5 family
    "gpt-5-2025-08-07-t1.0": "GPT-5 (reasoning)",
    "gpt-5-2025-08-07-no-reasoning-t1.0": "GPT-5",
    "gpt-5-mini-2025-08-07-t1.0": "GPT-5 Mini (reasoning)",
    "gpt-5-mini-2025-08-07-no-reasoning-t1.0": "GPT-5 Mini",

    # Qwen family
    "qwen3-next-80b-a3b-thinking-t1.0": "Qwen3-Next-80B (reasoning)",
    "qwen3-next-80b-a3b-instruct-t1.0": "Qwen3-Next-80B",

    # Claude family
    "claude-sonnet-4-20250514-t0.0": "Claude Sonnet 4 (reasoning)",
    "claude-sonnet-4-20250514-t1.0": "Claude Sonnet 4 (reasoning)",
    "claude-sonnet-4-20250514-no-reasoning-t0.0": "Claude Sonnet 4",
    "claude-sonnet-4-20250514-no-reasoning-t1.0": "Claude Sonnet 4",

    # DeepSeek family
    "deepseek-chat-v3.1-t1.0": "DeepSeek Chat v3.1",

    # LLaMA family
    "llama-3.3-70b-instruct-t1.0": "LLaMA-3.3-70B Instruct",
    "deepseek-r1-distill-llama-70b-t1.0": "DeepSeek R1-Distill LLaMA-70B",

    # Nemotron family
    "nemotron-nano-9b-v2-t1.0": "Nemotron-Nano 9B v2 (reasoning)",
    "nemotron-nano-9b-v2-no-reasoning-t1.0": "Nemotron-Nano 9B v2",

    # GPT-OSS
    "gpt-oss-120b-t1.0": "GPT-OSS 120B",
}

LANG_DIRS = ["results_en", "results_de", "results_it"]
GAME_NAME = "hot_air_balloon"
TARGET_TYPES = ("missing tag", "rule violation")  # print in this order

# -------- helpers --------
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
    if not s:
        return ""
    return s.replace("_", " ").strip().lower()

# --- model normalization + mapping ---
_model_keys = list(MODEL_NAME_MAP.keys())

_suffix_re = re.compile(r"-t\d+(?:\.\d+)?$")

def normalize_model_string(s: str) -> str:
    """basename, strip trailing '/', normalize case minimally, strip -tX[.Y] suffix."""
    base = os.path.basename(str(s).strip().rstrip("/"))
    # keep original case for exact match, but prepare a variant without -tX.Y
    return base

def strip_t_suffix(s: str) -> str:
    return _suffix_re.sub("", s)

def map_to_known_model(raw: str) -> str:
    """
    Try to map a raw results_folder string to a known MODEL_NAME_MAP key.
    Strategy:
      1) exact match
      2) match after stripping -tX[.Y]
      3) longest-prefix match vs known keys (on both raw and stripped forms)
    If none match, return the original raw.
    """
    candidate = normalize_model_string(raw)
    # 1) exact
    if candidate in MODEL_NAME_MAP:
        return candidate
    # 2) strip -t suffix and try again
    stripped = strip_t_suffix(candidate)
    if stripped in MODEL_NAME_MAP:
        return stripped
    # 3) longest-prefix match
    #    e.g., raw="deepseek-chat-v3.1" should map to key "deepseek-chat-v3.1-t1.0"
    best = None
    for key in _model_keys:
        if candidate.startswith(key) or key.startswith(candidate):
            if best is None or len(key) > len(best):
                best = key
        if stripped and (stripped.startswith(key) or key.startswith(stripped)):
            if best is None or len(key) > len(best):
                best = key
    return best if best else raw

def pretty_model_name(mapped_key_or_raw: str) -> str:
    return MODEL_NAME_MAP.get(mapped_key_or_raw, mapped_key_or_raw)

# -------- main --------
def main():
    repo_root = find_repo_root(Path(os.getcwd()))

    # Initialize counts with ALL mapped models so zeros show
    counts_by_key = {k: {t: 0 for t in TARGET_TYPES} for k in MODEL_NAME_MAP.keys()}
    extras = set()
    scanned_files = 0

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

            scanned_files += 1
            raw_folder = meta.get("results_folder") or "unknown_model"
            mapped_key = map_to_known_model(raw_folder)
            if mapped_key not in counts_by_key:
                counts_by_key.setdefault(mapped_key, {t: 0 for t in TARGET_TYPES})
                if mapped_key not in MODEL_NAME_MAP:
                    extras.add(mapped_key)

            for msg in flatten_turns(data.get("turns")):
                if msg.get("from") == "GM" and msg.get("to") == "GM":
                    a_type = norm_type((msg.get("action") or {}).get("type"))
                    if a_type in TARGET_TYPES:
                        key = "missing tag" if a_type == "missing tag" else "rule violation"
                        counts_by_key[mapped_key][key] += 1

    if scanned_files == 0:
        print("(no hot_air_balloon interactions.json files found under results_en/de/it)")
        return

    # Order: all mapped models (pretty sorted), then extras (pretty sorted)
    mapped_rows = []
    for key in MODEL_NAME_MAP.keys():
        c = counts_by_key.get(key, {t: 0 for t in TARGET_TYPES})
        mapped_rows.append((pretty_model_name(key), key, c["missing tag"], c["rule violation"]))
    mapped_rows.sort(key=lambda r: r[0].lower())

    extra_rows = []
    for key in sorted(extras, key=lambda k: pretty_model_name(k).lower()):
        c = counts_by_key.get(key, {t: 0 for t in TARGET_TYPES})
        extra_rows.append((pretty_model_name(key), key, c["missing tag"], c["rule violation"]))

    rows = mapped_rows + extra_rows

    # Print table
    name_col_width = max(len("Model"), max((len(r[0]) for r in rows), default=5))
    num_w = 14
    header = f"{'Model':<{name_col_width}}  {'missing tag':>{num_w}}  {'rule violation':>{num_w}}  {'total':>{num_w}}"
    print(header)
    print("-" * len(header))

    g_mt = g_rv = 0
    for pretty, _key, mt, rv in rows:
        total = mt + rv
        g_mt += mt
        g_rv += rv
        print(f"{pretty:<{name_col_width}}  {mt:>{num_w}d}  {rv:>{num_w}d}  {total:>{num_w}d}")

    print("-" * len(header))
    print(f"{'TOTAL':<{name_col_width}}  {g_mt:>{num_w}d}  {g_rv:>{num_w}d}  {g_mt + g_rv:>{num_w}d}")

if __name__ == "__main__":
    main()