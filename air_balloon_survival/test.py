import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Mapping raw model names to short readable names
MODEL_NAME_MAP = {
    "gpt-5-2025-08-07-t1.0": "GPT-5 (reasoning)",
    "gpt-5-2025-08-07-no-reasoning-t1.0": "GPT-5 (no reasoning)",
    "gpt-5-mini-2025-08-07-t1.0": "GPT-5 Mini (reasoning)",
    "gpt-5-mini-2025-08-07-no-reasoning-t1.0": "GPT-5 Mini (no reasoning)",
    "claude-sonnet-4-20250514-t0.0": "Claude Sonnet 4 (reasoning)",
    "claude-sonnet-4-20250514-t1.0": "Claude Sonnet 4 (reasoning)",  # alias → same as t0
    "claude-sonnet-4-20250514-no-reasoning-t0.0": "Claude Sonnet 4 (no reasoning)",
    "claude-sonnet-4-20250514-no-reasoning-t1.0": "Claude Sonnet 4 (no reasoning)",  # alias → same as t0
    "nemotron-nano-9b-v2-t1.0": "Nemotron-Nano 9B v2 (reasoning)",
    "nemotron-nano-9b-v2-no-reasoning-t1.0": "Nemotron-Nano 9B v2 (no reasoning)",
    "deepseek-r1-distill-llama-70b-t1.0": "DeepSeek R1-Distill LLaMA-70B",
    "llama-3.3-70b-instruct-t1.0": "LLaMA-3.3-70B Instruct",
}

def pretty_model_name(raw_name: str) -> str:
    return MODEL_NAME_MAP.get(raw_name, raw_name)  # fallback if unmapped


def collect_scores(base_paths):
    # experiment_type -> model_pretty -> { "score_sum": ..., "count": ..., "total_games": ... }
    data = defaultdict(lambda: defaultdict(lambda: {"score_sum": 0.0, "count": 0, "total_games": 0}))

    for base_path in base_paths:
        for model in os.listdir(base_path):
            model_path = os.path.join(base_path, model)
            if not os.path.isdir(model_path):
                continue

            hot_air_path = os.path.join(model_path, "hot_air_balloon")
            if not os.path.exists(hot_air_path):
                continue

            pretty_name = pretty_model_name(model)

            for experiment in os.listdir(hot_air_path):
                exp_path = os.path.join(hot_air_path, experiment)
                if not os.path.isdir(exp_path):
                    continue

                # experiment type = everything before "_easy"/"_hard"
                if experiment.endswith("_easy"):
                    exp_type = experiment[:-5]
                elif experiment.endswith("_hard"):
                    exp_type = experiment[:-5]
                else:
                    continue

                instances = [inst for inst in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, inst))]
                total_games = len(instances)

                for instance in instances:
                    inst_path = os.path.join(exp_path, instance)
                    summary_file = os.path.join(inst_path, "summary.json")
                    if not os.path.exists(summary_file):
                        continue

                    try:
                        with open(summary_file, "r") as f:
                            summary_data = json.load(f)

                        scores = summary_data.get("scores", {})
                        hm = scores.get("normalized_harmonic_mean")

                        if hm is not None:
                            data[exp_type][pretty_name]["score_sum"] += hm
                            data[exp_type][pretty_name]["count"] += 1

                    except Exception as e:
                        print(f"Failed to read {summary_file}: {e}")

                # update total games seen for that experiment type and model
                data[exp_type][pretty_name]["total_games"] += total_games

    return data

import numpy as np
import matplotlib.pyplot as plt

def plot_experiment_type_scores_grouped(data, output_file="experiment_scores_grouped.pdf"):
    # Substring → pretty label
    exp_types = {
        "complexity": "Complexity",
        "negotiation": "Negotiation",
        "reasoning": "No Reasoning Tag"
    }

    # Match actual keys in your data
    exp_keys = []
    for raw_key in data.keys():
        for substr, label in exp_types.items():
            if substr in raw_key:
                exp_keys.append((raw_key, label))
                break

    # Collect all models
    all_models = sorted({m for exp_key in data for m in data[exp_key]})

    # --- Compute global average score for sorting ---
    global_scores = {}
    for model in all_models:
        total_score, count = 0, 0
        for raw_key, _ in exp_keys:
            stats = data.get(raw_key, {}).get(model, {"score_sum": 0.0, "total_games": 0})
            total_games = stats["total_games"]
            score_sum = stats["score_sum"]
            if total_games > 0:
                total_score += score_sum / total_games
                count += 1
        global_scores[model] = total_score / count if count > 0 else 0

    # Sort models by global score (descending)
    all_models_sorted = sorted(all_models, key=lambda m: global_scores[m], reverse=True)

    # X positions
    x = np.arange(len(all_models_sorted))
    bar_width = 0.25

    plt.figure(figsize=(12, 6))
    colors = plt.cm.Set2.colors[:len(exp_keys)]

    for i, (raw_key, label) in enumerate(exp_keys):
        scores = []
        for model in all_models_sorted:
            stats = data.get(raw_key, {}).get(model, {"score_sum": 0.0, "total_games": 0})
            total_games = stats["total_games"]
            score_sum = stats["score_sum"]
            weighted_score = score_sum / total_games if total_games > 0 else 0
            scores.append(weighted_score)

        plt.bar(x + i * bar_width, scores, width=bar_width,
                label=label, color=colors[i])

    plt.xticks(x + bar_width, all_models_sorted, rotation=45, ha="right")
    plt.ylabel("Clemscore (Air Balloon Survival)")
    plt.title("Clemscore by Model and Experiment Type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":
    base_paths = {
        "en": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results_en")),
        "de": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results_de")),
    }

    for lang, path in base_paths.items():
        if not os.path.exists(path):
            print(f"Skipping {lang} (path not found: {path})")
            continue

        print(f"Processing {lang} results from {path}")
        data = collect_scores([path])
        plot_experiment_type_scores_grouped(
            data,
            output_file=f"experiment_scores_grouped_{lang}.pdf"
        )
