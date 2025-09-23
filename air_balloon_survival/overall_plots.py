import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def collect_scores(base_paths):
    results = defaultdict(lambda: {"hard": [], "easy": []})

    for base_path in base_paths:
        for model in os.listdir(base_path):
            model_path = os.path.join(base_path, model, "hot_air_balloon")
            if not os.path.isdir(model_path):
                continue

            for experiment in os.listdir(model_path):
                exp_path = os.path.join(model_path, experiment)
                if not os.path.isdir(exp_path):
                    continue

                difficulty = None
                if experiment.endswith("hard"):
                    difficulty = "hard"
                elif experiment.endswith("easy"):
                    difficulty = "easy"
                else:
                    continue

                for instance in os.listdir(exp_path):
                    inst_path = os.path.join(exp_path, instance)
                    summary_file = os.path.join(inst_path, "summary.json")
                    if not os.path.exists(summary_file):
                        continue

                    try:
                        with open(summary_file, "r") as f:
                            summary = json.load(f)

                        agreement = summary.get("agreement")
                        if isinstance(agreement, dict):
                            score = agreement.get("normalized_harmonic_mean")
                            if score is not None:
                                results[model][difficulty].append(score)
                            else:
                                results[model][difficulty].append(0.0)
                        else:
                            results[model][difficulty].append(0.0)

                    except Exception as e:
                        print(f"Failed to read {summary_file}: {e}")

    return results


def plot_weighted_scores(results, difficulty, out_file):
    models = []
    weighted_scores = []

    for model, vals in results.items():
        scores = vals[difficulty]
        if scores:
            weighted = np.mean(scores)  # average, with missing as 0
            models.append(model)
            weighted_scores.append(weighted)

    # order by performance
    sorted_pairs = sorted(zip(models, weighted_scores), key=lambda x: x[1], reverse=True)
    models, weighted_scores = zip(*sorted_pairs)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, weighted_scores)
    plt.ylabel("Average main score (0 if unplayed)")
    plt.title(f"Main Score Averaged Over All Instances ({difficulty})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_file, format="pdf")
    plt.close()


if __name__ == "__main__":
    base_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results_en")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results_de")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results_it"))
    ]

    results = collect_scores(base_paths)

    plot_weighted_scores(results, "hard", "weighted_scores_hard.pdf")
    plot_weighted_scores(results, "easy", "weighted_scores_easy.pdf")