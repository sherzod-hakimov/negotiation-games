import os
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

FOCUS_EXPERIMENTS = {
    "air_balloon_survival_en_negotiation_hard",
    "air_balloon_survival_en_reasoning off_hard",
}


def compute_model_metrics(base_path: str):
    model_results = {}
    for model in os.listdir(base_path):
        model_path = os.path.join(base_path, model)
        if not os.path.isdir(model_path):
            continue

        hot_air_path = os.path.join(model_path, "hot_air_balloon")
        if not os.path.exists(hot_air_path):
            continue

        # containers
        adherence_rates = []
        alternation_rates = []
        final_u1_vals, final_u2_vals = [], []
        final_u1_vals_focus, final_u2_vals_focus = [], []
        stubborn_p1, stubborn_p2, stubborn_total = [], [], []
        stubborn_p1_focus, stubborn_p2_focus, stubborn_total_focus = [], [], []

        per_idx_changes = defaultdict(list)
        per_idx_main_scores = defaultdict(list)
        per_idx_changes_focus = defaultdict(list)
        per_idx_diff_focus = defaultdict(list)
        per_idx_scores_focus = defaultdict(list)

        for experiment in os.listdir(hot_air_path):
            exp_path = os.path.join(hot_air_path, experiment)
            if not os.path.isdir(exp_path):
                continue

            for instance in os.listdir(exp_path):
                inst_path = os.path.join(exp_path, instance)
                if not os.path.isdir(inst_path):
                    continue

                summary_file = os.path.join(inst_path, "summary.json")
                if not os.path.exists(summary_file):
                    continue

                try:
                    with open(summary_file, "r") as f:
                        summary_data = json.load(f)

                    scores = summary_data.get("scores", {})
                    agreement = summary_data.get("agreement")

                    # final normalized utils
                    if isinstance(agreement, dict):
                        u1 = agreement.get("normalized_u1")
                        u2 = agreement.get("normalized_u2")
                        if u1 is not None:
                            final_u1_vals.append(u1)
                        if u2 is not None:
                            final_u2_vals.append(u2)
                        if experiment in FOCUS_EXPERIMENTS:
                            if u1 is not None:
                                final_u1_vals_focus.append(u1)
                            if u2 is not None:
                                final_u2_vals_focus.append(u2)

                    # stubbornness
                    st1 = scores.get("stubbornness_player1")
                    st2 = scores.get("stubbornness_player2")
                    stt = scores.get("stubbornness_total")
                    if st1 is not None:
                        stubborn_p1.append(st1)
                    if st2 is not None:
                        stubborn_p2.append(st2)
                    if stt is not None:
                        stubborn_total.append(stt)
                    if experiment in FOCUS_EXPERIMENTS:
                        if st1 is not None:
                            stubborn_p1_focus.append(st1)
                        if st2 is not None:
                            stubborn_p2_focus.append(st2)
                        if stt is not None:
                            stubborn_total_focus.append(stt)

                    # pareto / alternation
                    pareto_rate = scores.get("pareto_adherence_rate")
                    if pareto_rate is not None:
                        adherence_rates.append(pareto_rate)
                    alternation_rate = scores.get("alternation_rate")
                    if alternation_rate is not None:
                        alternation_rates.append(alternation_rate)

                    # per-proposal changes
                    prop_changes = summary_data.get("normalized_substitutions_per_proposal", [])
                    for idx, val in enumerate(prop_changes, start=1):  # start at proposal 2
                        if val is not None:
                            per_idx_changes[idx].append(val)
                            if experiment in FOCUS_EXPERIMENTS:
                                per_idx_changes_focus[idx].append(val)

                    proposals = summary_data.get("proposals", [])
                    for idx, proposal in enumerate(proposals, start=1):
                        val = proposal.get("normalized_harmonic_mean")
                        if val is not None:
                            per_idx_main_scores[idx].append(val)
                            if experiment in FOCUS_EXPERIMENTS:
                                per_idx_scores_focus[idx].append(val)

                        if experiment in FOCUS_EXPERIMENTS:
                            u1 = proposal.get("normalized_u1")
                            u2 = proposal.get("normalized_u2")
                            if u1 is not None and u2 is not None:
                                per_idx_diff_focus[idx].append(abs(u1 - u2))

                except Exception as e:
                    print(f"Failed to read {summary_file}: {e}")

        # store averages (empty lists → None, partial lists → valid avg)
        def safe_mean(vals):
            return float(np.mean(vals)) if vals else None

        model_results[model] = {
            "avg_pareto_adherence_rate": safe_mean(adherence_rates),
            "avg_alternation_rate": safe_mean(alternation_rates),
            "avg_stubbornness_player1": safe_mean(stubborn_p1),
            "avg_stubbornness_player2": safe_mean(stubborn_p2),
            "avg_stubbornness_total": safe_mean(stubborn_total),
            "avg_per_idx_changes": {idx: safe_mean(vals) for idx, vals in per_idx_changes.items()},
            "avg_per_idx_main_scores": {idx: safe_mean(vals) for idx, vals in per_idx_main_scores.items()},
            "avg_final_normalized_u1": safe_mean(final_u1_vals),
            "avg_final_normalized_u2": safe_mean(final_u2_vals),
            # focus
            "avg_per_idx_changes_focus": {idx: safe_mean(vals) for idx, vals in per_idx_changes_focus.items()},
            "avg_per_idx_diff_focus": {idx: safe_mean(vals) for idx, vals in per_idx_diff_focus.items()},
            "avg_per_idx_scores_focus": {idx: safe_mean(vals) for idx, vals in per_idx_scores_focus.items()},
            "avg_stubbornness_player1_focus": safe_mean(stubborn_p1_focus),
            "avg_stubbornness_player2_focus": safe_mean(stubborn_p2_focus),
            "avg_stubbornness_total_focus": safe_mean(stubborn_total_focus),
            "avg_final_normalized_u1_focus": safe_mean(final_u1_vals_focus),
            "avg_final_normalized_u2_focus": safe_mean(final_u2_vals_focus),
        }

    return model_results


def plot_changes(results, max_points=12):
    """Plot normalized substitutions (all experiments) per model."""
    for model, metrics in results.items():
        idx_changes = metrics["avg_per_idx_changes"]
        if not idx_changes:
            continue

        xs = sorted(idx_changes.keys())[:max_points]
        ys = [idx_changes[x] for x in xs]

        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, linewidth=1.5, marker="o", markersize=3)

        display_name = MODEL_NAME_MAP.get(model, model)   # use mapped name
        plt.xlabel("Proposal index")
        plt.ylabel("Avg. normalized substitutions")
        plt.title(f"Substitutions over proposals in temporal order\n{display_name}")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        # save using mapped name (cleaner filenames)
        safe_model = display_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(f"air_balloon_survival/substitutions_all_{safe_model}.pdf")
        plt.close()

MODEL_NAME_MAP = {
    # GPT-5 family
    "gpt-5-2025-08-07-t1.0": "GPT-5 (reasoning)",
    "gpt-5-2025-08-07-no-reasoning-t1.0": "GPT-5",
    "gpt-5-mini-2025-08-07-t1.0": "GPT-5 Mini (reasoning)",
    "gpt-5-mini-2025-08-07-no-reasoning-t1.0": "GPT-5 Mini",
    "gpt-oss-120b-t1.0": "GPT-OSS 120B",

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
    "deepseek-r1-distill-llama-70b-t1.0": "DeepSeek R1-Distill LLaMA-70B (reasoning)",

    # Nemotron family
    "nemotron-nano-9b-v2-t1.0": "Nemotron-Nano 9B v2 (reasoning)",
    "nemotron-nano-9b-v2-no-reasoning-t1.0": "Nemotron-Nano 9B v2"
}


def plot_focus_stubbornness(results):
    """Plot stubbornness (focus experiments) with two bars per model (P1, P2), sorted by total stubbornness."""
    models = []
    p1_vals = []
    p2_vals = []
    totals = []

    for model, metrics in results.items():
        s1 = metrics.get("avg_stubbornness_player1_focus")
        s2 = metrics.get("avg_stubbornness_player2_focus")
        st = metrics.get("avg_stubbornness_total_focus")
        if s1 is not None and s2 is not None and st is not None:
            models.append(MODEL_NAME_MAP.get(model, model))
            p1_vals.append(s1)
            p2_vals.append(s2)
            totals.append(st)

    if not models:
        return

    # Sort by total stubbornness (descending)
    sorted_indices = np.argsort(totals)[::-1]
    models = [models[i] for i in sorted_indices]
    p1_vals = [p1_vals[i] for i in sorted_indices]
    p2_vals = [p2_vals[i] for i in sorted_indices]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, p1_vals, width, label="Player 1")
    plt.bar(x + width/2, p2_vals, width, label="Player 2")

    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel("Avg. Stubbornness (Focus)")
    plt.title("Average Stubbornness by Player (Focus Experiments)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("air_balloon_survival/stubbornness_focus.pdf")
    plt.close()

REASONING_MODELS = {
    "gpt-5-2025-08-07-t1.0",
    "gpt-5-mini-2025-08-07-t1.0",
    "claude-sonnet-4-20250514-t0.0",
    "claude-sonnet-4-20250514-t1.0",
    "nemotron-nano-9b-v2-t1.0",
    "deepseek-r1-distill-llama-70b-t1.0",
}

NONREASONING_MODELS = {
    "gpt-5-2025-08-07-no-reasoning-t1.0",
    "gpt-5-mini-2025-08-07-no-reasoning-t1.0",
    "claude-sonnet-4-20250514-no-reasoning-t0.0",
    "claude-sonnet-4-20250514-no-reasoning-t1.0",
    "nemotron-nano-9b-v2-no-reasoning-t1.0",
    "llama-3.3-70b-instruct-t1.0",
}
def plot_focus_stubbornness_split(results):
    """Plot stubbornness (focus experiments) separately for reasoning and non-reasoning models."""

    reasoning_models, reasoning_p1, reasoning_p2, reasoning_totals = [], [], [], []
    nonreasoning_models, nonreasoning_p1, nonreasoning_p2, nonreasoning_totals = [], [], [], []

    for model, metrics in results.items():
        s1 = metrics.get("avg_stubbornness_player1_focus")
        s2 = metrics.get("avg_stubbornness_player2_focus")
        st = metrics.get("avg_stubbornness_total_focus")
        if None in (s1, s2, st):
            continue

        name = MODEL_NAME_MAP.get(model, model)

        if model in REASONING_MODELS:
            reasoning_models.append(name)
            reasoning_p1.append(s1)
            reasoning_p2.append(s2)
            reasoning_totals.append(st)
        elif model in NONREASONING_MODELS:
            nonreasoning_models.append(name)
            nonreasoning_p1.append(s1)
            nonreasoning_p2.append(s2)
            nonreasoning_totals.append(st)

    def plot_group(models, p1_vals, p2_vals, totals, title, filename):
        if not models:
            return
        # sort by total stubbornness
        sorted_idx = np.argsort(totals)[::-1]
        models = [models[i] for i in sorted_idx]
        p1_vals = [p1_vals[i] for i in sorted_idx]
        p2_vals = [p2_vals[i] for i in sorted_idx]

        x = np.arange(len(models))
        width = 0.35

        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, p1_vals, width, label="Player 1")
        plt.bar(x + width/2, p2_vals, width, label="Player 2")
        plt.xticks(x, models, rotation=45, ha="right")
        plt.ylabel("Avg. Stubbornness")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    plot_group(reasoning_models, reasoning_p1, reasoning_p2, reasoning_totals,
               "Average Stubbornness (Reasoning Models, Opposing Goals)",
               "air_balloon_survival/stubbornness_focus_reasoning.pdf")

    plot_group(nonreasoning_models, nonreasoning_p1, nonreasoning_p2, nonreasoning_totals,
               "Average Stubbornness (Non-Reasoning Models, Opposing Goals)",
               "air_balloon_survival/stubbornness_focus_nonreasoning.pdf")

def plot_focus_scores_split(results):
    """Plot final normalized scores (focus experiments) separately for reasoning and non-reasoning models,
    ordered the same way as the stubbornness plots (per group)."""

    reasoning_models, reasoning_u1, reasoning_u2, reasoning_totals = [], [], [], []
    nonreasoning_models, nonreasoning_u1, nonreasoning_u2, nonreasoning_totals = [], [], [], []

    for model, metrics in results.items():
        u1 = metrics.get("avg_final_normalized_u1_focus")
        u2 = metrics.get("avg_final_normalized_u2_focus")
        st = metrics.get("avg_stubbornness_total_focus")
        if None in (u1, u2, st):
            continue

        name = MODEL_NAME_MAP.get(model, model)

        if model in REASONING_MODELS:
            reasoning_models.append(name)
            reasoning_u1.append(u1)
            reasoning_u2.append(u2)
            reasoning_totals.append(st)
        elif model in NONREASONING_MODELS:
            nonreasoning_models.append(name)
            nonreasoning_u1.append(u1)
            nonreasoning_u2.append(u2)
            nonreasoning_totals.append(st)

    def plot_group(models, u1_vals, u2_vals, totals, title, filename):
        if not models:
            return
        # sort by stubbornness totals
        sorted_idx = np.argsort(totals)[::-1]
        models = [models[i] for i in sorted_idx]
        u1_vals = [u1_vals[i] for i in sorted_idx]
        u2_vals = [u2_vals[i] for i in sorted_idx]

        x = np.arange(len(models))
        width = 0.35

        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, u1_vals, width, label="Player 1")
        plt.bar(x + width/2, u2_vals, width, label="Player 2")
        plt.xticks(x, models, rotation=45, ha="right")
        plt.ylabel("Avg. Final Normalized Utility (Focus)")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    plot_group(reasoning_models, reasoning_u1, reasoning_u2, reasoning_totals,
               "Average Final Scores (Reasoning Models, Focus Experiments)",
               "air_balloon_survival/final_scores_focus_reasoning.pdf")

    plot_group(nonreasoning_models, nonreasoning_u1, nonreasoning_u2, nonreasoning_totals,
               "Average Final Scores (Non-Reasoning Models, Focus Experiments)",
               "air_balloon_survival/final_scores_focus_nonreasoning.pdf")

# Define GPT-family models (both reasoning and no-reasoning, including Mini)
GPT_MODELS = {
    "gpt-5-2025-08-07-t1.0",
    "gpt-5-2025-08-07-no-reasoning-t1.0",
    "gpt-5-mini-2025-08-07-t1.0",
    "gpt-5-mini-2025-08-07-no-reasoning-t1.0",
    "gpt-oss-120b-t1.0"
}

def plot_focus_scores_gpt(results):
    """Plot final normalized scores (focus experiments) for GPT-family models only,
    ordered by stubbornness totals."""

    models, u1_vals, u2_vals, totals = [], [], [], []

    for model, metrics in results.items():
        if model not in GPT_MODELS:
            continue  # skip non-GPT models

        u1 = metrics.get("avg_final_normalized_u1_focus")
        u2 = metrics.get("avg_final_normalized_u2_focus")
        st = metrics.get("avg_stubbornness_total_focus")

        if None in (u1, u2, st):
            continue

        models.append(MODEL_NAME_MAP.get(model, model))
        u1_vals.append(u1)
        u2_vals.append(u2)
        totals.append(st)

    if not models:
        return

    # sort by stubbornness totals
    sorted_idx = np.argsort(totals)[::-1]
    models = [models[i] for i in sorted_idx]
    u1_vals = [u1_vals[i] for i in sorted_idx]
    u2_vals = [u2_vals[i] for i in sorted_idx]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, u1_vals, width, label="Player 1")
    plt.bar(x + width/2, u2_vals, width, label="Player 2")
    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel("Avg. Normalized Player Score")
    plt.title("Average Player Scores for GPT-family Models (Opposing Goals)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("air_balloon_survival/final_scores_focus_gpt.pdf")
    plt.close()



def plot_focus_stubbornness_gpt(results):
    """Plot stubbornness (focus experiments) for GPT-family models only,
    ordered by total stubbornness."""

    models, p1_vals, p2_vals, totals = [], [], [], []

    for model, metrics in results.items():
        if model not in GPT_MODELS:
            continue

        s1 = metrics.get("avg_stubbornness_player1_focus")
        s2 = metrics.get("avg_stubbornness_player2_focus")
        st = metrics.get("avg_stubbornness_total_focus")
        if None in (s1, s2, st):
            continue

        models.append(MODEL_NAME_MAP.get(model, model))
        p1_vals.append(s1)
        p2_vals.append(s2)
        totals.append(st)

    if not models:
        return

    # sort by stubbornness totals
    sorted_idx = np.argsort(totals)[::-1]
    models = [models[i] for i in sorted_idx]
    p1_vals = [p1_vals[i] for i in sorted_idx]
    p2_vals = [p2_vals[i] for i in sorted_idx]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, p1_vals, width, label="Player 1")
    plt.bar(x + width/2, p2_vals, width, label="Player 2")
    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel("Avg. Stubbornness")
    plt.title("Average Stubbornness by Players (Opposing Goals)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("air_balloon_survival/stubbornness_focus_model_choice.pdf")
    plt.close()

def compute_all_languages(base_root: str, lang_dirs=("results_en", "results_fr", "results_pt")):
    """Aggregate results across multiple languages by averaging metrics per model."""
    from collections import defaultdict

    all_results = defaultdict(list)

    for lang in lang_dirs:
        path = os.path.join(base_root, lang)
        if not os.path.exists(path):
            continue
        res = compute_model_metrics(path)
        for model, metrics in res.items():
            all_results[model].append(metrics)

    averaged = {}
    for model, metrics_list in all_results.items():
        averaged[model] = {}
        keys = metrics_list[0].keys()
        for key in keys:
            vals = []
            if isinstance(metrics_list[0][key], dict):
                # dicts: average per index
                idx_all = defaultdict(list)
                for m in metrics_list:
                    for idx, v in m[key].items():
                        if v is not None:
                            idx_all[idx].append(v)
                averaged[model][key] = {idx: float(np.mean(vs)) for idx, vs in idx_all.items()}
            else:
                # scalars: just average over languages
                for m in metrics_list:
                    val = m.get(key)
                    if val is not None:
                        vals.append(val)
                averaged[model][key] = float(np.mean(vals)) if vals else None
    return averaged

def plot_focus_changes(results, max_points=12):
    """Plot normalized substitutions (focus experiments only) per model."""
    for model, metrics in results.items():
        idx_changes_focus = metrics["avg_per_idx_changes_focus"]
        if not idx_changes_focus:
            continue

        xs = sorted(idx_changes_focus.keys())[:max_points]
        ys = [idx_changes_focus[x] for x in xs]

        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, linewidth=1.5, marker="o", markersize=3)

        display_name = MODEL_NAME_MAP.get(model, model)
        plt.xlabel("Proposal index")
        plt.ylabel("Avg. normalized substitutions (focus only)")
        plt.title(f"Substitutions over proposals (focus experiments)\n{display_name}")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        safe_model = display_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(f"air_balloon_survival/substitutions_focus_{safe_model}.pdf")
        plt.close()

if __name__ == "__main__":
    base_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    lang_dirs = ["results_en", "results_de", "results_it"]

    log_path = os.path.join(base_root, "results_summary.log")
    with open(log_path, "w", encoding="utf-8") as log:

        def log_print(*args, **kwargs):
            """Helper to write both to file and (optionally) stdout"""
            msg = " ".join(str(a) for a in args)
            log.write(msg + "\n")

        log_print("=== Per-language results ===")
        for lang in lang_dirs:
            path = os.path.join(base_root, lang)
            if not os.path.exists(path):
                continue

            log_print(f"\n### Language: {lang} ###")
            results = compute_model_metrics(path)

            log_print("\n=== Average Pareto adherence rate per model ===")
            for model, metrics in results.items():
                pareto = metrics["avg_pareto_adherence_rate"]
                pareto_str = f"{pareto:.3f}" if pareto is not None else "no data"
                log_print(f"{model}: {pareto_str}")

            log_print("\n=== Average alternation rate per model ===")
            for model, metrics in results.items():
                alternation = metrics["avg_alternation_rate"]
                alternation_str = f"{alternation:.3f}" if alternation is not None else "no data"
                log_print(f"{model}: {alternation_str}")

            log_print("\n=== Average stubbornness (all experiments) ===")
            for model, metrics in results.items():
                s1 = metrics["avg_stubbornness_player1"]
                s2 = metrics["avg_stubbornness_player2"]
                st = metrics["avg_stubbornness_total"]
                if None not in (s1, s2, st):
                    log_print(f"{model}: P1={s1:.3f} P2={s2:.3f} Total={st:.3f}")
                else:
                    log_print(f"{model}: no data")

            log_print("\n=== Average stubbornness (focus experiments) ===")
            for model, metrics in results.items():
                s1 = metrics["avg_stubbornness_player1_focus"]
                s2 = metrics["avg_stubbornness_player2_focus"]
                st = metrics["avg_stubbornness_total_focus"]
                if None not in (s1, s2, st):
                    log_print(f"{model}: P1={s1:.3f} P2={s2:.3f} Total={st:.3f}")
                else:
                    log_print(f"{model}: no data")

            log_print("\n=== Average final normalized utilities (all experiments) ===")
            for model, metrics in results.items():
                u1 = metrics["avg_final_normalized_u1"]
                u2 = metrics["avg_final_normalized_u2"]
                if None not in (u1, u2):
                    log_print(f"{model}: P1={u1:.3f} P2={u2:.3f}")
                else:
                    log_print(f"{model}: no data")

            log_print("\n=== Average final normalized utilities (focus experiments) ===")
            for model, metrics in results.items():
                u1 = metrics["avg_final_normalized_u1_focus"]
                u2 = metrics["avg_final_normalized_u2_focus"]
                if None not in (u1, u2):
                    log_print(f"{model}: P1={u1:.3f} P2={u2:.3f}")
                else:
                    log_print(f"{model}: no data")

            # plots for this language
            plot_changes(results)
            plot_focus_stubbornness_gpt(results)
            plot_focus_scores_gpt(results)

        # === cross-language average ===
        log_print("\n\n=== Cross-language averaged results ===")
        results = compute_all_languages(base_root, lang_dirs=lang_dirs)

        log_print("\n=== Average Pareto adherence rate per model ===")
        for model, metrics in results.items():
            pareto = metrics["avg_pareto_adherence_rate"]
            pareto_str = f"{pareto:.3f}" if pareto is not None else "no data"
            log_print(f"{model}: {pareto_str}")

        log_print("\n=== Average alternation rate per model ===")
        for model, metrics in results.items():
            alternation = metrics["avg_alternation_rate"]
            alternation_str = f"{alternation:.3f}" if alternation is not None else "no data"
            log_print(f"{model}: {alternation_str}")

        log_print("\n=== Average stubbornness (all experiments) ===")
        for model, metrics in results.items():
            s1 = metrics["avg_stubbornness_player1"]
            s2 = metrics["avg_stubbornness_player2"]
            st = metrics["avg_stubbornness_total"]
            if None not in (s1, s2, st):
                log_print(f"{model}: P1={s1:.3f} P2={s2:.3f} Total={st:.3f}")
            else:
                log_print(f"{model}: no data")

        log_print("\n=== Average stubbornness (focus experiments) ===")
        for model, metrics in results.items():
            s1 = metrics["avg_stubbornness_player1_focus"]
            s2 = metrics["avg_stubbornness_player2_focus"]
            st = metrics["avg_stubbornness_total_focus"]
            if None not in (s1, s2, st):
                log_print(f"{model}: P1={s1:.3f} P2={s2:.3f} Total={st:.3f}")
            else:
                log_print(f"{model}: no data")

        log_print("\n=== Average final normalized utilities (all experiments) ===")
        for model, metrics in results.items():
            u1 = metrics["avg_final_normalized_u1"]
            u2 = metrics["avg_final_normalized_u2"]
            if None not in (u1, u2):
                log_print(f"{model}: P1={u1:.3f} P2={u2:.3f}")
            else:
                log_print(f"{model}: no data")

        log_print("\n=== Average final normalized utilities (focus experiments) ===")
        for model, metrics in results.items():
            u1 = metrics["avg_final_normalized_u1_focus"]
            u2 = metrics["avg_final_normalized_u2_focus"]
            if None not in (u1, u2):
                log_print(f"{model}: P1={u1:.3f} P2={u2:.3f}")
            else:
                log_print(f"{model}: no data")

        # plots for averaged results
        plot_changes(results)
        plot_focus_stubbornness_gpt(results)
        plot_focus_scores_gpt(results)
        plot_focus_changes(results)

    print(f"\nLog written to {log_path}")