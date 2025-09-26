import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# -----------------------
# Model pretty names
# -----------------------
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
    "deepseek-chat-v3.1-t1.0": "DeepSeek Chat v3.1 (reasoning)",

    # LLaMA family
    "llama-3.3-70b-instruct-t1.0": "LLaMA-3.3-70B Instruct",
    "deepseek-r1-distill-llama-70b-t1.0": "DeepSeek R1-Distill LLaMA-70B (reasoning)",

    # Nemotron family
    "nemotron-nano-9b-v2-t1.0": "Nemotron-Nano 9B v2 (reasoning)",
    "nemotron-nano-9b-v2-no-reasoning-t1.0": "Nemotron-Nano 9B v2",

    # GPT-OSS
    "gpt-oss-120b-t1.0": "GPT-OSS 120B (reasoning)",
}
def pretty_model_name(raw: str) -> str:
    return MODEL_NAME_MAP.get(raw, raw)

# -----------------------
# Bucket rules per game
# -----------------------
def bucket_hot_air_balloon(exp_name: str):
    e = exp_name.lower()
    if e.endswith("easy"):
        return "Easy"
    if e.endswith("hard"):
        return "Hard"
    return None

def bucket_clean_up(exp_name: str):
    e = exp_name.lower()
    if "easy" in e:
        return "Easy"
    if "medium" in e:
        return "Medium"
    if "hard" in e:
        return "Hard"
    return None

def bucket_dond(exp_name: str):
    e = exp_name.lower()
    if e.startswith("coop"):
        return "Coop"
    if e.startswith("semi"):
        return "Semi"
    return None

BUCKET_FN = {
    "hot_air_balloon": bucket_hot_air_balloon,
    "clean_up":        bucket_clean_up,
    "dond":            bucket_dond,
}

GAME_CONFIG = {
    "hot_air_balloon": {
        "order": ["Easy", "Hard"],
        "colors": {"Easy": "#66c2a5", "Hard": "#fc8d62"},
        "ylabel": "Clemscore (Easy/Hard)"
    },
    "clean_up": {
        "order": ["Easy", "Medium", "Hard"],
        "colors": {"Easy": "#66c2a5", "Medium": "#8da0cb", "Hard": "#fc8d62"},
        "ylabel": "Clemscore (Easy/Medium/Hard)"
    },
    "dond": {
        # Coop first, Semi second
        "order": ["Coop", "Semi"],
        # Match Easy/Hard colors
        "colors": {"Coop": "#66c2a5", "Semi": "#fc8d62"},
        "ylabel": "Clemscore (Coop/Semi)"
    },
}

# -----------------------
# Audit helpers
# -----------------------
def audit_non_numeric_for_game(df_game: pd.DataFrame, game: str, lang: str, outdir: str):
    """
    Prints and saves a report of entries for metrics {Success, Main Score}
    whose 'value' cannot be parsed as a number. Does not mutate df_game.
    """
    if df_game.empty:
        return

    bucket_func = BUCKET_FN[game]
    df = df_game.copy()
    df["bucket"] = df["experiment"].astype(str).map(bucket_func)
    df = df[df["metric"].isin(["Success", "Main Score"])].copy()
    if df.empty:
        return

    df["value_raw"] = df["value"]
    coerced = pd.to_numeric(df["value"], errors="coerce")
    bad_mask = df["value"].notna() & coerced.isna()
    bad = df.loc[bad_mask, ["game", "bucket", "experiment", "episode", "model", "metric", "value_raw"]].copy()

    def reason(s: str) -> str:
        if s is None:
            return "None"
        st = str(s).strip()
        if st == "":
            return "empty string"
        low = st.lower()
        if low in ("true", "false", "yes", "no"):
            return "boolean string"
        if st.endswith("%"):
            return "percent string"
        if re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", st) and not st.isnumeric():
            return "contains number + extra text"
        return "non-numeric string"

    if not bad.empty:
        bad["reason"] = bad["value_raw"].map(reason)
        print(f"\n[AUDIT] Non-numeric 'value' entries for {game} — {lang.upper()}: {len(bad)} rows")
        summary = (
            bad.groupby(["metric", "reason"])
               .size()
               .rename("count")
               .reset_index()
               .sort_values("count", ascending=False)
        )
        print(summary.to_string(index=False))

        print("\n[Examples]")
        print(bad.head(20).to_string(index=False))

        out_dir = Path(outdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"non_numeric_values_{game}_{lang}.csv"
        bad.to_csv(out_path, index=False)
        print(f"[AUDIT] Full list saved to: {out_path}\n")
    else:
        print(f"[AUDIT] No non-numeric 'value' entries for {game} — {lang.upper()}.")

# -----------------------
# Numeric coercion (for plotting step)
# -----------------------
def coerce_value_to_float(x):
    """Robust numeric coercion for 'value' column. NaNs stay NaN so pandas ignores them in mean()."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)

    s = str(x).strip()
    sl = s.lower()
    if sl in ("true", "yes"):
        return 1.0
    if sl in ("false", "no"):
        return 0.0

    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return np.nan

    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return np.nan

    return np.nan

# -----------------------
# Core compute
# -----------------------
def compute_bucket_products(df_game: pd.DataFrame, game: str) -> pd.DataFrame:
    """
    For a single game, compute (avg Success) * (avg Main Score) per model per bucket.
    Returns a wide DF: index=model_pretty, columns=buckets, values=product.
    """
    if df_game.empty:
        return pd.DataFrame()

    bucket_func = BUCKET_FN[game]
    df_g = df_game.copy()
    df_g["bucket"] = df_g["experiment"].astype(str).map(bucket_func)
    df_g = df_g[~df_g["bucket"].isna()]
    if df_g.empty:
        return pd.DataFrame()

    df_g["model_pretty"] = df_g["model"].map(pretty_model_name)

    # Keep only the two metrics we average
    df_g = df_g[df_g["metric"].isin(["Success", "Main Score"])].copy()
    if df_g.empty:
        return pd.DataFrame()

    # Convert to numeric; NaNs remain NaN and are ignored by mean()
    df_g["value"] = df_g["value"].apply(coerce_value_to_float)

    # Average value by (model_pretty, bucket, metric), over ALL rows (episodes etc.)
    grouped = (
        df_g.groupby(["model_pretty", "bucket", "metric"], dropna=False)["value"]
            .mean()
            .reset_index()
    )

    # Pivot to align metrics side by side
    pivot = grouped.pivot_table(
        index=["model_pretty", "bucket"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()

    # Fill missing after averaging so product is defined
    pivot["Success"] = pivot["Success"].fillna(0.0)
    pivot["Main Score"] = pivot["Main Score"].fillna(0.0)
    pivot["product"] = pivot["Success"] * pivot["Main Score"]

    # Wide format: models x buckets
    wide = pivot.pivot_table(
        index="model_pretty",
        columns="bucket",
        values="product",
        aggfunc="mean"
    ).fillna(0.0)

    # Stable column order per game
    desired = GAME_CONFIG[game]["order"]
    cols = [c for c in desired if c in wide.columns] + [c for c in wide.columns if c not in desired]
    return wide[cols] if not wide.empty else wide

def sort_models_by_global_avg(wide: pd.DataFrame, bucket_order: list[str]) -> list[str]:
    """
    Compute a global average across the listed buckets and return models sorted desc.
    """
    if wide.empty:
        return []
    cols = [c for c in bucket_order if c in wide.columns]
    if not cols:
        return list(wide.index)
    global_avg = wide[cols].mean(axis=1)
    return list(global_avg.sort_values(ascending=False).index)

# -----------------------
# Plotting
# -----------------------
def plot_grouped_bars(wide: pd.DataFrame, game: str, lang: str, outdir: str):
    """
    One chart per game per language. Bars grouped by difficulty bucket.
    """
    if wide.empty:
        print(f"[{lang}] {game}: no data to plot.")
        return

    cfg = GAME_CONFIG[game]
    order = cfg["order"]
    colors = cfg["colors"]

    models_sorted = sort_models_by_global_avg(wide, order)
    x = np.arange(len(models_sorted))
    n_buckets = len(order)
    bar_width = 0.8 / max(n_buckets, 1)

    plt.figure(figsize=(max(10, 1.2 * len(models_sorted)), 6))
    handles = []
    labels = []

    for i, bucket in enumerate(order):
        vals = wide.loc[models_sorted, bucket].values if bucket in wide.columns else np.zeros(len(models_sorted))
        bars = plt.bar(x + i * bar_width, vals, width=bar_width, label=bucket, color=colors.get(bucket))
        handles.append(bars[0])
        labels.append(bucket)

    plt.xticks(x + (n_buckets - 1) * bar_width / 2, models_sorted, rotation=45, ha="right")
    plt.ylabel(cfg["ylabel"])
    pretty_game = {"hot_air_balloon":"Hot Air Balloon", "clean_up":"Clean Up", "dond":"DoND"}.get(game, game)
    plt.title(f"{pretty_game} — {lang.upper()}")
    plt.legend(handles, labels)
    plt.tight_layout()

    out_path = Path(outdir) / f"{game}_{lang}.pdf"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

# -----------------------
# Repo root detection
# -----------------------
def find_repo_root(start: Path) -> Path:
    """
    Prefer CWD; if it doesn't contain results_* with raw.csv, walk up until we find one.
    """
    p = start.resolve()
    candidates = ["results_en", "results_de", "results_it"]
    while True:
        for c in candidates:
            if (p / c / "raw.csv").exists():
                return p
        if p.parent == p:
            return start.resolve()
        p = p.parent

# -----------------------
# Driver
# -----------------------
def process_language(results_dir: str, lang: str, outdir: str = "plots"):
    """
    results_dir: path to results_<lang> containing raw.csv
    """
    csv_path = os.path.join(results_dir, "raw.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping {lang}: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    required = {"game", "experiment", "episode", "metric", "value", "model"}
    missing = required - set(df.columns)
    if missing:
        print(f"Skipping {lang}: raw.csv missing columns {missing}")
        return

    # Per-game processing: audit THEN compute/plot
    for game in ["hot_air_balloon", "clean_up", "dond"]:
        df_game = df[df["game"] == game].copy()

        # 🔎 Audit exact problematic rows (saved under results_<lang>/audits/)
        audits_dir = os.path.join(results_dir, "audits")
        audit_non_numeric_for_game(df_game, game, lang, outdir=audits_dir)

        wide = compute_bucket_products(df_game, game)
        plot_grouped_bars(wide, game, lang, outdir)

if __name__ == "__main__":
    # Use current working directory as the starting point
    cwd = Path(os.getcwd())
    repo_root = find_repo_root(cwd)

    lang_dirs = {
        "en": repo_root / "results_en",
        "de": repo_root / "results_de",
        "it": repo_root / "results_it",
    }

    outdir = repo_root / "bar_plots_game_scores_grouped"
    outdir.mkdir(parents=True, exist_ok=True)

    for lang, path in lang_dirs.items():
        print(f"Processing {lang} from {path}")
        process_language(str(path), lang, outdir=str(outdir))