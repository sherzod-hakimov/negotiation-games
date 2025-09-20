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

        adherence_rates = []
        alternation_rates = []
        per_idx_changes = defaultdict(list)
        per_idx_main_scores = defaultdict(list)
        conv_lengths = []

        # Focus experiment only
        per_idx_changes_focus = defaultdict(list)
        per_idx_diff_focus = defaultdict(list)

        for experiment in os.listdir(hot_air_path):
            exp_path = os.path.join(hot_air_path, experiment)
            if not os.path.isdir(exp_path):
                continue

            # --- regular stats for all experiments ---
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
                    proposals = summary_data.get("proposals", [])
                    conv_len = len(proposals)
                    conv_lengths.append(conv_len)

                    pareto_rate = scores.get("pareto_adherence_rate", None)
                    if pareto_rate is not None:
                        adherence_rates.append(pareto_rate)

                    alternation_rate = scores.get("alternation_rate", None)
                    if alternation_rate is not None:
                        alternation_rates.append(alternation_rate)

                    # per-proposal substitutions
                    prop_changes = summary_data.get("normalized_substitutions_per_proposal", [])
                    for idx, val in enumerate(prop_changes, start=1):
                        if val is not None:
                            per_idx_changes[idx].append(val)

                    # per-proposal harmonic mean
                    for idx, proposal in enumerate(proposals, start=1):
                        val = proposal.get("normalized_harmonic_mean", None)
                        if val is not None:
                            per_idx_main_scores[idx].append(val)

                    # --- focus experiments only ---
                    if experiment in FOCUS_EXPERIMENTS:
                        # substitutions
                        for idx, val in enumerate(prop_changes, start=1):
                            if val is not None:
                                per_idx_changes_focus[idx].append(val)

                        # abs diff between players
                        for idx, proposal in enumerate(proposals, start=1):
                            u1 = proposal.get("normalized_u1")
                            u2 = proposal.get("normalized_u2")
                            if u1 is not None and u2 is not None:
                                per_idx_diff_focus[idx].append(abs(u1 - u2))

                except Exception as e:
                    print(f"Failed to read {summary_file}: {e}")

        model_results[model] = {
            # all experiments
            "avg_pareto_adherence_rate": float(np.mean(adherence_rates)) if adherence_rates else None,
            "avg_alternation_rate": float(np.mean(alternation_rates)) if alternation_rates else None,
            "avg_per_idx_changes": {idx: float(np.mean(vals)) for idx, vals in per_idx_changes.items()},
            "avg_per_idx_main_scores": {idx: float(np.mean(vals)) for idx, vals in per_idx_main_scores.items()},
            # focus only
            "avg_per_idx_changes_focus": {idx: float(np.mean(vals)) for idx, vals in per_idx_changes_focus.items()},
            "avg_per_idx_diff_focus": {idx: float(np.mean(vals)) for idx, vals in per_idx_diff_focus.items()},
        }

    return model_results


def plot_focus_changes(results):
    """Plot normalized substitutions (focus experiments) per model."""
    for model, metrics in results.items():
        idx_changes_focus = metrics["avg_per_idx_changes_focus"]
        if not idx_changes_focus:
            continue

        xs = sorted(idx_changes_focus.keys())
        ys = [idx_changes_focus[x] for x in xs]

        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, linewidth=1.5, marker="o", markersize=3)
        plt.xlabel("Proposal index")
        plt.ylabel("Avg. normalized substitutions")
        plt.title(f"Normalized substitutions per proposal\n{model} (focus experiments)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        # save one pdf per model
        safe_model = model.replace("/", "_")
        plt.savefig(f"substitutions_focus_{safe_model}.pdf")
        plt.close()


if __name__ == "__main__":
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results_en"))
    results = compute_model_metrics(base_path)

    print("\n=== Average Pareto adherence rate per model ===")
    for model, metrics in results.items():
        pareto = metrics["avg_pareto_adherence_rate"]
        pareto_str = f"{pareto:.3f}" if pareto is not None else "no data"
        print(f"{model}: {pareto_str}")

    print("\n=== Average alternation rate per model ===")
    for model, metrics in results.items():
        alternation = metrics["avg_alternation_rate"]
        alternation_str = f"{alternation:.3f}" if alternation is not None else "no data"
        print(f"{model}: {alternation_str}")

    print("\n=== Average normalized substitutions per proposal index (all experiments) ===")
    for model, metrics in results.items():
        idx_changes = metrics["avg_per_idx_changes"]
        if not idx_changes:
            print(f"{model}: no data")
        else:
            idx_str = ", ".join([f"idx {idx}: {val:.3f}" for idx, val in sorted(idx_changes.items())])
            print(f"{model}: {idx_str}")

    print("\n=== Average normalized harmonic mean per proposal index (all experiments) ===")
    for model, metrics in results.items():
        idx_scores = metrics["avg_per_idx_main_scores"]
        if not idx_scores:
            print(f"{model}: no data")
        else:
            idx_str = ", ".join([f"idx {idx}: {val:.3f}" for idx, val in sorted(idx_scores.items())])
            print(f"{model}: {idx_str}")

    print("\n=== Focus experiments only (normalized substitutions per proposal index) ===")
    for model, metrics in results.items():
        idx_changes_focus = metrics["avg_per_idx_changes_focus"]
        if not idx_changes_focus:
            print(f"{model}: no data")
        else:
            idx_str = ", ".join([f"idx {idx}: {val:.3f}" for idx, val in sorted(idx_changes_focus.items())])
            print(f"{model}: {idx_str}")

    print("\n=== Focus experiments only (absolute difference between players per proposal index) ===")
    for model, metrics in results.items():
        idx_diff_focus = metrics["avg_per_idx_diff_focus"]
        if not idx_diff_focus:
            print(f"{model}: no data")
        else:
            idx_str = ", ".join([f"idx {idx}: {val:.3f}" for idx, val in sorted(idx_diff_focus.items())])
            print(f"{model}: {idx_str}")

    # plot focus-only substitutions per model
    plot_focus_changes(results)