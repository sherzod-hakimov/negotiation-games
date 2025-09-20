import os
import json
import numpy as np
from collections import defaultdict

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

        # For binning
        all_instances = []

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
                    proposals = summary_data.get("proposals", [])
                    conv_len = len(proposals)
                    conv_lengths.append(conv_len)

                    pareto_rate = scores.get("pareto_adherence_rate", None)
                    if pareto_rate is not None:
                        adherence_rates.append(pareto_rate)

                    alternation_rate = scores.get("alternation_rate", None)
                    if alternation_rate is not None:
                        alternation_rates.append(alternation_rate)

                    # Collect per-proposal substitution ratios
                    prop_changes = summary_data.get("normalized_substitutions_per_proposal", None)
                    if prop_changes:
                        for idx, val in enumerate(prop_changes, start=1):
                            if val is not None:
                                per_idx_changes[idx].append(val)

                    # Collect per-proposal main scores
                    for idx, proposal in enumerate(proposals, start=1):
                        val = proposal.get("normalized_harmonic_mean", None)
                        if val is not None:
                            per_idx_main_scores[idx].append(val)

                    # Store instance-level data for later binning
                    all_instances.append((conv_len, prop_changes, [p.get("normalized_harmonic_mean") for p in proposals]))

                except Exception as e:
                    print(f"Failed to read {summary_file}: {e}")

        # Compute quantiles for binning
        bins = None
        if conv_lengths:
            q1, q2 = np.percentile(conv_lengths, [33, 66])
            bins = (q1, q2)

        # Assign instances to bins
        binned_changes = {"short": defaultdict(list), "medium": defaultdict(list), "long": defaultdict(list)}
        binned_scores = {"short": defaultdict(list), "medium": defaultdict(list), "long": defaultdict(list)}

        if bins:
            q1, q2 = bins
            for conv_len, changes, scores_list in all_instances:
                if conv_len <= q1:
                    bin_name = "short"
                elif conv_len <= q2:
                    bin_name = "medium"
                else:
                    bin_name = "long"

                if changes:
                    for idx, val in enumerate(changes, start=1):
                        if val is not None:
                            binned_changes[bin_name][idx].append(val)

                if scores_list:
                    for idx, val in enumerate(scores_list, start=1):
                        if val is not None:
                            binned_scores[bin_name][idx].append(val)

        model_results[model] = {
            "avg_pareto_adherence_rate": float(np.mean(adherence_rates)) if adherence_rates else None,
            "avg_alternation_rate": float(np.mean(alternation_rates)) if alternation_rates else None,
            "avg_per_idx_changes": {idx: float(np.mean(vals)) for idx, vals in per_idx_changes.items()},
            "avg_per_idx_main_scores": {idx: float(np.mean(vals)) for idx, vals in per_idx_main_scores.items()},
            "avg_per_idx_changes_binned": {
                bin_name: {idx: float(np.mean(vals)) for idx, vals in idx_dict.items()}
                for bin_name, idx_dict in binned_changes.items()
            },
            "avg_per_idx_main_scores_binned": {
                bin_name: {idx: float(np.mean(vals)) for idx, vals in idx_dict.items()}
                for bin_name, idx_dict in binned_scores.items()
            },
            "length_bins": bins,
        }

    return model_results


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

    print("\n=== Average normalized substitutions per proposal index ===")
    for model, metrics in results.items():
        idx_changes = metrics["avg_per_idx_changes"]
        if not idx_changes:
            print(f"{model}: no data")
        else:
            idx_str = ", ".join([f"idx {idx}: {val:.3f}" for idx, val in sorted(idx_changes.items())])
            print(f"{model}: {idx_str}")

    print("\n=== Average normalized harmonic mean per proposal index ===")
    for model, metrics in results.items():
        idx_scores = metrics["avg_per_idx_main_scores"]
        if not idx_scores:
            print(f"{model}: no data")
        else:
            idx_str = ", ".join([f"idx {idx}: {val:.3f}" for idx, val in sorted(idx_scores.items())])
            print(f"{model}: {idx_str}")

    print("\n=== Average normalized substitutions per proposal index (binned) ===")
    for model, metrics in results.items():
        idx_changes_binned = metrics["avg_per_idx_changes_binned"]
        if not any(idx_changes_binned.values()):
            print(f"{model}: no data")
        else:
            for bin_name, idx_dict in idx_changes_binned.items():
                if not idx_dict:
                    continue
                idx_str = ", ".join([f"idx {idx}: {val:.3f}" for idx, val in sorted(idx_dict.items())])
                print(f"{model} ({bin_name}): {idx_str}")

    print("\n=== Average normalized harmonic mean per proposal index (binned) ===")
    for model, metrics in results.items():
        main_scores_binned = metrics["avg_per_idx_main_scores_binned"]
        if not any(main_scores_binned.values()):
            print(f"{model}: no data")
        else:
            for bin_name, idx_dict in main_scores_binned.items():
                if not idx_dict:
                    continue
                idx_str = ", ".join([f"idx {idx}: {val:.3f}" for idx, val in sorted(idx_dict.items())])
                print(f"{model} ({bin_name}): {idx_str}")

    print("\n=== Conversation length bins per model ===")
    for model, metrics in results.items():
        bins = metrics.get("length_bins", None)
        if bins:
            q1, q2 = bins
            print(f"{model}: short ≤ {q1:.1f}, medium ≤ {q2:.1f}, long > {q2:.1f}")
        else:
            print(f"{model}: no length data")