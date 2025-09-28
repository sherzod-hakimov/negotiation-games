import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from adjustText import adjust_text

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
        "order": ["Coop", "Semi"],
        "colors": {"Coop": "#66c2a5", "Semi": "#fc8d62"},
        "ylabel": "Clemscore (Coop/Semi)"
    },
}

# -----------------------
# Audit helpers
# -----------------------
def audit_non_numeric_for_game(df_game: pd.DataFrame, game: str, lang: str, outdir: str):
    """Report entries for metrics {Success, Main Score} whose 'value' is non-numeric; save details to CSV."""
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
        out_dir = Path(outdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"non_numeric_values_{game}_{lang}.csv"
        bad.to_csv(out_path, index=False)
        print(f"[AUDIT] Full list saved to: {out_path}\n")
    else:
        print(f"[AUDIT] No non-numeric 'value' entries for {game} — {lang.upper()}.")

# -----------------------
# Numeric coercion
# -----------------------
def coerce_value_to_float(x):
    """Coerce heterogeneous values to float; leave invalids as NaN."""
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
# Core compute for bar plots
# -----------------------
def compute_bucket_products(df_game: pd.DataFrame, game: str) -> pd.DataFrame:
    """For a single game, compute (avg Success) * (avg Main Score) per model per bucket; return wide table."""
    if df_game.empty:
        return pd.DataFrame()
    bucket_func = BUCKET_FN[game]
    df_g = df_game.copy()
    df_g["bucket"] = df_g["experiment"].astype(str).map(bucket_func)
    df_g = df_g[~df_g["bucket"].isna()]
    if df_g.empty:
        return pd.DataFrame()
    df_g["model_pretty"] = df_g["model"].map(pretty_model_name)
    df_g = df_g[df_g["metric"].isin(["Success", "Main Score"])].copy()
    if df_g.empty:
        return pd.DataFrame()
    df_g["value"] = df_g["value"].apply(coerce_value_to_float)

    grouped = (
        df_g.groupby(["model_pretty", "bucket", "metric"], dropna=False)["value"]
            .mean()
            .reset_index()
    )

    pivot = grouped.pivot_table(
        index=["model_pretty", "bucket"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()

    pivot["Success"] = pivot["Success"].fillna(0.0)
    pivot["Main Score"] = pivot["Main Score"].fillna(0.0)
    pivot["product"] = pivot["Success"] * pivot["Main Score"]

    wide = pivot.pivot_table(
        index="model_pretty",
        columns="bucket",
        values="product",
        aggfunc="mean"
    ).fillna(0.0)

    desired = GAME_CONFIG[game]["order"]
    cols = [c for c in desired if c in wide.columns] + [c for c in wide.columns if c not in desired]
    return wide[cols] if not wide.empty else wide

def sort_models_by_global_avg(wide: pd.DataFrame, bucket_order: list[str]) -> list[str]:
    """Sort models by the average across the given bucket columns."""
    if wide.empty:
        return []
    cols = [c for c in bucket_order if c in wide.columns]
    if not cols:
        return list(wide.index)
    global_avg = wide[cols].mean(axis=1)
    return list(global_avg.sort_values(ascending=False).index)

# -----------------------
# Bar plotting
# -----------------------
def plot_grouped_bars(wide: pd.DataFrame, game: str, lang: str, outdir: str):
    """Grouped bar charts per game and language."""
    if wide.empty:
        print(f"[{lang}] {game}: no data to plot.")
        return

    FONTS = {"title": 25, "label": 20, "ticks": 18, "legend": 16}
    DPI = 200
    cfg = GAME_CONFIG[game]
    order = cfg["order"]
    colors = cfg["colors"]

    models_sorted = sort_models_by_global_avg(wide, order)
    x = np.arange(len(models_sorted))
    n_buckets = len(order)
    bar_width = 0.8 / max(n_buckets, 1)

    fig_width = max(10, 1.2 * len(models_sorted))
    fig_height = 10
    plt.figure(figsize=(fig_width, fig_height), dpi=DPI)

    handles, labels = [], []
    for i, bucket in enumerate(order):
        vals = wide.loc[models_sorted, bucket].values if bucket in wide.columns else np.zeros(len(models_sorted))
        bars = plt.bar(x + i * bar_width, vals, width=bar_width, label=bucket, color=colors.get(bucket))
        handles.append(bars[0]); labels.append(bucket)

    plt.xticks(x + (n_buckets - 1) * bar_width / 2, models_sorted, rotation=45, ha="right", fontsize=FONTS["ticks"])
    plt.yticks(fontsize=FONTS["ticks"])
    plt.ylabel(cfg["ylabel"], fontsize=FONTS["label"])

    pretty_game = {"hot_air_balloon": "Air Balloon Survival", "clean_up": "Clean Up", "dond": "DoND"}.get(game, game)
    plt.title(f"{pretty_game} — {lang.upper()}", fontsize=FONTS["title"])
    plt.legend(handles, labels, fontsize=FONTS["legend"])

    plt.tight_layout()
    out_path = Path(outdir) / f"{game}_{lang}.pdf"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

# -----------------------
# Repo root detection
# -----------------------
def find_repo_root(start: Path) -> Path:
    """Walk up from 'start' until we find a results_<lang>/raw.csv, else return start."""
    p = start.resolve()
    candidates = ["results_en", "results_de", "results_it"]
    while True:
        for c in candidates:
            if (p / c / "raw.csv").exists():
                return p
        if p.parent == p:
            return start.resolve()
        p = p.parent

# =====================================================
# Scatter data + plots (overall results per language)
# =====================================================

SCATTER_RESULTS: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)

ACRONYM_MAP = {
    "GPT-5 (reasoning)": "G5R",
    "GPT-5": "G5",
    "GPT-5 Mini (reasoning)": "G5mR",
    "GPT-5 Mini": "G5m",
    "Qwen3-Next-80B (reasoning)": "Q3R",
    "Qwen3-Next-80B": "Q3",
    "Claude Sonnet 4 (reasoning)": "C4R",
    "Claude Sonnet 4": "C4",
    "DeepSeek Chat v3.1 (reasoning)": "DS3R",
    "LLaMA-3.3-70B Instruct": "L3",
    "DeepSeek R1-Distill LLaMA-70B (reasoning)": "DSL3",
    "Nemotron-Nano 9B v2 (reasoning)": "NN9R",
    "Nemotron-Nano 9B v2": "NN9",
    "GPT-OSS 120B (reasoning)": "GOR",
}
def model_acronym(raw_model: str) -> str:
    pretty = pretty_model_name(raw_model)
    if pretty in ACRONYM_MAP:
        return ACRONYM_MAP[pretty]
    return re.sub(r'[^A-Za-z0-9]+', '', pretty).upper()[:5]

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

def compute_overall_xy_for_game(df_game: pd.DataFrame) -> pd.DataFrame:
    """Per game/model: x = avg(Main Score), y = avg(Played Games)."""
    if df_game.empty:
        return pd.DataFrame(columns=["model_pretty", "acronym", "x", "y"])
    df = df_game.copy()
    df["value"] = df["value"].apply(coerce_value_to_float)
    df["model_pretty"] = df["model"].map(pretty_model_name)
    df["acronym"] = df["model"].map(model_acronym)

    df_x = (df[df["metric"] == "Main Score"]
              .groupby("model_pretty", as_index=False)["value"].mean()
              .rename(columns={"value": "x"}))

    played_metric_names = _select_played_metric_names(df)
    if played_metric_names:
        df_y = (df[df["metric"].isin(played_metric_names)]
                  .groupby("model_pretty", as_index=False)["value"].mean()
                  .rename(columns={"value": "y"}))
    else:
        tmp = (df.groupby(["model_pretty", "experiment"])["episode"]
                 .nunique()
                 .groupby("model_pretty").mean()
                 .rename("y").reset_index())
        df_y = tmp

    merged = pd.merge(df_x, df_y, on="model_pretty", how="outer").fillna(0.0)
    acr = df[["model_pretty", "acronym"]].drop_duplicates()
    merged = pd.merge(merged, acr, on="model_pretty", how="left")
    return merged[["model_pretty", "acronym", "x", "y"]].sort_values("x", ascending=False)

def compute_overall_xy_across_games(df_lang: pd.DataFrame) -> pd.DataFrame:
    """Per model across games: x = avg(Main Score), y = avg(Played Games)."""
    if df_lang.empty:
        return pd.DataFrame(columns=["model_pretty", "acronym", "x", "y"])
    df = df_lang.copy()
    df["value"] = df["value"].apply(coerce_value_to_float)
    df["model_pretty"] = df["model"].map(pretty_model_name)
    df["acronym"] = df["model"].map(model_acronym)

    x_df = (df[df["metric"] == "Main Score"]
              .groupby("model_pretty", as_index=False)["value"].mean()
              .rename(columns={"value": "x"}))

    played_metric_names = _select_played_metric_names(df)
    if played_metric_names:
        y_df = (df[df["metric"].isin(played_metric_names)]
                  .groupby("model_pretty", as_index=False)["value"].mean()
                  .rename(columns={"value": "y"}))
    else:
        tmp = (df.groupby(["model_pretty", "game", "experiment"])["episode"]
                 .nunique()
                 .groupby("model_pretty").mean()
                 .rename("y").reset_index())
        y_df = tmp

    merged = pd.merge(x_df, y_df, on="model_pretty", how="outer").fillna(0.0)
    acr = df[["model_pretty", "acronym"]].drop_duplicates()
    merged = pd.merge(merged, acr, on="model_pretty", how="left")
    return merged[["model_pretty", "acronym", "x", "y"]].sort_values("x", ascending=False)

def _model_color_map(df_lang: pd.DataFrame) -> dict:
    """Stable color assignment per model for a given language."""
    models = sorted(pd.Series(df_lang["model"].unique(), dtype=str).map(pretty_model_name))
    cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['#1f77b4'])
    colors = {}
    for i, m in enumerate(models):
        colors[m] = cycle[i % len(cycle)]
    return colors

def _plot_scatter(df_xy: pd.DataFrame, title: str, outfile: Path, model_colors: dict, ann_fs: int = 7):
    """Scatter with per-model colors and auto-adjusted text using adjustText."""

    FONTS = {"axis": 16, "ticks": 12, "title": 18}
    DPI = 200
    fig, ax = plt.subplots(1, 1, figsize=(7.8, 6.6), dpi=DPI)

    # Axes and grid
    ax.set_title(title, fontsize=FONTS["title"])
    ax.set_xlabel("% Played", fontsize=FONTS["axis"])       # X = played (%)
    ax.set_ylabel("Quality Score", fontsize=FONTS["axis"])  # Y = quality
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=FONTS["ticks"])

    # Empty case
    if df_xy.empty:
        plt.tight_layout()
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, bbox_inches="tight")
        plt.close(fig)
        return

    # Scatter points (stable color per model)
    colors = [model_colors.get(m) for m in df_xy["model_pretty"]]
    x = df_xy["y"].to_numpy() * 100  # Played, scaled to %
    y = df_xy["x"].to_numpy()        # Quality
    ax.scatter(x, y, s=40, alpha=0.9, c=colors)

    # Limits before label adjustment
    xmax, ymax = float(x.max()), float(y.max())
    ax.set_xlim(left=0.0, right=max(100.0, xmax * 1.05))
    ax.set_ylim(bottom=0.0, top=max(1.0, ymax * 1.10))

    # Create text objects at the points (small font)
    texts = [
        ax.text(float(xi), float(yi), str(lbl),
                fontsize=ann_fs, ha="center", va="center", zorder=3)
        for xi, yi, lbl in zip(x, y, df_xy["acronym"])
    ]

    # Adjust text positions to avoid overlaps
    fig.canvas.draw()
    adjust_text(
        texts,
        x=x,
        y=y,
        ax=ax,
        autoalign="xy",
        only_move={"points": "", "text": "xy"},
        expand_points=(1.9, 2.5),
        expand_text=(1.6, 1.75),
        force_points=2,
        force_text=1.10,
        lim=800
    )

    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

def plot_scatter_per_game(df_lang: pd.DataFrame, lang: str, scatter_outdir: str):
    """Create one PDF per (language, game) with all models; also export CSV per game."""
    games = ["hot_air_balloon", "clean_up", "dond"]
    titles = {"hot_air_balloon": "Air Balloon Survival", "clean_up": "Clean Up", "dond": "DoND"}

    scatter_dir = Path(scatter_outdir)
    scatter_dir.mkdir(parents=True, exist_ok=True)

    color_map = _model_color_map(df_lang)

    for game in games:
        gdf = df_lang[df_lang["game"] == game].copy()
        xy = compute_overall_xy_for_game(gdf)
        SCATTER_RESULTS[lang][game] = xy.copy()
        xy.to_csv(scatter_dir / f"overall_xy_{lang}_{game}.csv", index=False)

        title = f"{titles.get(game, game)} — {lang.upper()}"
        outfile = scatter_dir / f"scatter_{lang}_{game}.pdf"
        _plot_scatter(xy, title, outfile, model_colors=color_map, ann_fs=7)

def plot_scatter_across_games(df_lang: pd.DataFrame, lang: str, scatter_outdir: str):
    """Create one PDF per language with averages across games; also export combined CSV."""
    scatter_dir = Path(scatter_outdir)
    scatter_dir.mkdir(parents=True, exist_ok=True)

    color_map = _model_color_map(df_lang)
    xy_all = compute_overall_xy_across_games(df_lang)
    xy_all.to_csv(scatter_dir / f"overall_xy_{lang}.csv", index=False)

    title = f"Overall (across games) — {lang.upper()}"
    outfile = scatter_dir / f"overall_scatter_across_games_{lang}.pdf"
    _plot_scatter(xy_all, title, outfile, model_colors=color_map, ann_fs=7)

# -----------------------
# Driver
# -----------------------
def process_language(results_dir: str, lang: str, bar_outdir: str = "plots/bars", scatter_outdir: str = "plots/scatter"):
    """Produce grouped bar charts and scatter plots (per game + across games) for a language."""
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

    # Audits + grouped bar charts
    for game in ["hot_air_balloon", "clean_up", "dond"]:
        df_game = df[df["game"] == game].copy()
        audits_dir = os.path.join(results_dir, "audits")
        audit_non_numeric_for_game(df_game, game, lang, outdir=audits_dir)

        wide = compute_bucket_products(df_game, game)
        plot_grouped_bars(wide, game, lang, bar_outdir)

    # Scatter plots: one PDF per game, plus one overall PDF across games
    plot_scatter_per_game(df, lang, scatter_outdir)
    plot_scatter_across_games(df, lang, scatter_outdir)

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    cwd = Path(os.getcwd())
    repo_root = find_repo_root(cwd)

    lang_dirs = {
        "en": repo_root / "results_en",
        "de": repo_root / "results_de",
        "it": repo_root / "results_it",
    }

    bar_outdir = repo_root / "bar_plots_game_scores_grouped"
    scatter_outdir = repo_root / "overall_results_scatter_plots"

    bar_outdir.mkdir(parents=True, exist_ok=True)
    scatter_outdir.mkdir(parents=True, exist_ok=True)

    for lang, path in lang_dirs.items():
        print(f"Processing {lang} from {path}")
        process_language(str(path), lang, bar_outdir=str(bar_outdir), scatter_outdir=str(scatter_outdir))