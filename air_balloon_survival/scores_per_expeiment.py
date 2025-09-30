import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------
# Model pretty names
# -----------------------
MODEL_NAME_MAP = {
    "gpt-5-2025-08-07-t1.0": "GPT-5 (reasoning)",
    "gpt-5-2025-08-07-no-reasoning-t1.0": "GPT-5",
    "gpt-5-mini-2025-08-07-t1.0": "GPT-5 Mini (reasoning)",
    "gpt-5-mini-2025-08-07-no-reasoning-t1.0": "GPT-5 Mini",

    "qwen3-next-80b-a3b-thinking-t1.0": "Qwen3-Next-80B (reasoning)",
    "qwen3-next-80b-a3b-instruct-t1.0": "Qwen3-Next-80B",

    "claude-sonnet-4-20250514-t0.0": "Claude Sonnet 4 (reasoning)",
    "claude-sonnet-4-20250514-t1.0": "Claude Sonnet 4 (reasoning)",
    "claude-sonnet-4-20250514-no-reasoning-t0.0": "Claude Sonnet 4",
    "claude-sonnet-4-20250514-no-reasoning-t1.0": "Claude Sonnet 4",

    "deepseek-chat-v3.1-t1.0": "DeepSeek Chat v3.1 (reasoning)",

    "llama-3.3-70b-instruct-t1.0": "LLaMA-3.3-70B Instruct",
    "deepseek-r1-distill-llama-70b-t1.0": "DeepSeek R1-Distill LLaMA-70B (reasoning)",

    "nemotron-nano-9b-v2-t1.0": "Nemotron-Nano 9B v2 (reasoning)",
    "nemotron-nano-9b-v2-no-reasoning-t1.0": "Nemotron-Nano 9B v2",

    "gpt-oss-120b-t1.0": "GPT-OSS 120B (reasoning)",
}

def pretty_model_name(raw: str) -> str:
    return MODEL_NAME_MAP.get(raw, raw)


# -----------------------
# Acronyms
# -----------------------
ACRONYM_MAP = {
    "GPT-5 (reasoning)": "G5R",
    "GPT-5": "G5",
    "GPT-5 Mini (reasoning)": "G5mR",
    "GPT-5 Mini": "G5m",
    "Qwen3-Next-80B (reasoning)": "Q80R",
    "Qwen3-Next-80B": "Q80",
    "Claude Sonnet 4 (reasoning)": "CS4R",
    "Claude Sonnet 4": "CS4",
    "DeepSeek Chat v3.1 (reasoning)": "DS3R",
    "LLaMA-3.3-70B Instruct": "L70",
    "DeepSeek R1-Distill LLaMA-70B (reasoning)": "R1D70",
    "Nemotron-Nano 9B v2 (reasoning)": "NN9R",
    "Nemotron-Nano 9B v2": "NN9",
    "GPT-OSS 120B (reasoning)": "O120R",
}

def model_acronym(raw_model: str) -> str:
    pretty = pretty_model_name(raw_model)
    return ACRONYM_MAP.get(pretty, re.sub(r'[^A-Za-z0-9]+', '', pretty).upper()[:5])


# -----------------------
# Numeric coercion
# -----------------------
def coerce_value_to_float(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().lower()
    if s in ("true", "yes"): return 1.0
    if s in ("false", "no"): return 0.0
    if s.endswith("%"):
        try: return float(s[:-1]) / 100.0
        except ValueError: return np.nan
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if m:
        try: return float(m.group(0))
        except ValueError: return np.nan
    return np.nan


# -----------------------
# Played metric detection
# -----------------------
PLAYED_GAMES_CANDIDATES = [
    "Average Played Games", "Avg Played Games", "Played Games",
    "AverageGamesPlayed", "AvgGamesPlayed", "Games Played",
    "Played_Game_Avg", "Avg#Games", "Avg Episodes Played"
]
_PLAYED_REGEX = re.compile(r"(avg|average)?\s*.*(played|games|episodes).*", re.IGNORECASE)

def _select_played_metric_names(df: pd.DataFrame) -> list[str]:
    present = set(df["metric"].astype(str).unique())
    exact = [m for m in PLAYED_GAMES_CANDIDATES if m in present]
    if exact:
        return exact
    return [m for m in present if _PLAYED_REGEX.fullmatch(m) or _PLAYED_REGEX.search(m)]


# -----------------------
# Extract per-experiment scores (hot_air_balloon only)
# -----------------------
def extract_scores_and_played(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["game"] == "hot_air_balloon"].copy()
    if df.empty:
        return pd.DataFrame(columns=["game", "experiment", "model_pretty", "acronym", "main_score", "played_rate"])

    df["value"] = df["value"].apply(coerce_value_to_float)
    df["model_pretty"] = df["model"].map(pretty_model_name)
    df["acronym"] = df["model"].map(model_acronym)

    # Main Score
    df_main = (
        df[df["metric"] == "Main Score"]
        .groupby(["game", "experiment", "model_pretty", "acronym"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "main_score"})
    )

    # Played rate
    played_metric_names = _select_played_metric_names(df)
    if played_metric_names:
        df_played = (
            df[df["metric"].isin(played_metric_names)]
            .groupby(["game", "experiment", "model_pretty", "acronym"], as_index=False)["value"]
            .mean()
            .rename(columns={"value": "played_rate"})
        )
    else:
        tmp = (
            df.groupby(["game", "experiment", "model_pretty", "acronym"])["episode"]
            .nunique()
            .rename("played_rate")
            .reset_index()
        )
        df_played = tmp

    merged = pd.merge(df_main, df_played, on=["game", "experiment", "model_pretty", "acronym"], how="outer").fillna(0.0)
    return merged


# -----------------------
# Compute deltas (Easy – Hard)
# -----------------------
def compute_deltas(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(columns=["experiment_base", "model_pretty", "acronym", "delta_main_score", "delta_played_rate"])

    scores = scores.copy()

    # Split experiment into base + difficulty
    def split_base(exp):
        e = str(exp).lower()
        if e.endswith("easy"): return exp[:-4].rstrip(), "easy"
        if e.endswith("hard"): return exp[:-4].rstrip(), "hard"
        return exp, None

    tmp = scores.copy()
    tmp[["experiment_base", "difficulty"]] = tmp["experiment"].apply(lambda x: pd.Series(split_base(x)))

    # Pivot to wide format: columns easy/hard for both metrics
    wide = tmp.pivot_table(
        index=["experiment_base", "model_pretty", "acronym"],
        columns="difficulty",
        values=["main_score", "played_rate"],
        aggfunc="mean"
    )

    wide = wide.reset_index()
    # Compute deltas
    wide["delta_main_score"] = wide["main_score"]["easy"] - wide["main_score"]["hard"]
    wide["delta_played_rate"] = wide["played_rate"]["easy"] - wide["played_rate"]["hard"]

    return wide[["experiment_base", "model_pretty", "acronym", "delta_main_score", "delta_played_rate"]]


# -----------------------
# Driver
# -----------------------
def main():
    cwd = Path(os.getcwd())
    repo_root = cwd.resolve()

    lang_dirs = {
        "en": repo_root / "results_en",
        "de": repo_root / "results_de",
        "it": repo_root / "results_it",
    }

    outdir = repo_root / "per_experiment_scores"
    outdir.mkdir(parents=True, exist_ok=True)

    for lang, path in lang_dirs.items():
        csv_path = path / "raw.csv"
        if not csv_path.exists():
            print(f"Skipping {lang}: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        required = {"game", "experiment", "episode", "metric", "value", "model"}
        missing = required - set(df.columns)
        if missing:
            print(f"Skipping {lang}: raw.csv missing columns {missing}")
            continue

        scores = extract_scores_and_played(df)
        outfile_scores = outdir / f"per_experiment_scores_{lang}.csv"
        scores.to_csv(outfile_scores, index=False)
        print(f"Saved {outfile_scores}")

        deltas = compute_deltas(scores)
        outfile_deltas = outdir / f"per_experiment_deltas_{lang}.csv"
        deltas.to_csv(outfile_deltas, index=False)
        print(f"Saved {outfile_deltas}")


if __name__ == "__main__":
    main()