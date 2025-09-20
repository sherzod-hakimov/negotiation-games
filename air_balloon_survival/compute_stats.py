import os
import json
import numpy as np

def compute_model_pareto_adherence(base_path: str):
    model_results = {}

    for model in os.listdir(base_path):
        model_path = os.path.join(base_path, model)
        if not os.path.isdir(model_path):
            continue

        hot_air_path = os.path.join(model_path, "hot_air_balloon")
        if not os.path.exists(hot_air_path):
            continue

        adherence_rates = []

        for experiment in os.listdir(hot_air_path):
            exp_path = os.path.join(hot_air_path, experiment)
            if not os.path.isdir(exp_path):
                continue

            for instance in os.listdir(exp_path):
                inst_path = os.path.join(exp_path, instance)
                if not os.path.isdir(inst_path):
                    continue

                summary_file = os.path.join(inst_path, "summary.json")
                if os.path.exists(summary_file):
                    try:
                        with open(summary_file, "r") as f:
                            summary_data = json.load(f)

                        rate = summary_data.get("scores", {}).get("pareto_adherence_rate", None)
                        if rate is not None:
                            adherence_rates.append(rate)
                    except Exception as e:
                        print(f"Failed to read {summary_file}: {e}")

        if adherence_rates:
            model_results[model] = float(np.mean(adherence_rates))
        else:
            model_results[model] = None  # no valid data

    return model_results


if __name__ == "__main__":
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results_en"))
    results = compute_model_pareto_adherence(base_path)

    print("\nAverage Pareto adherence rate per model:")
    for model, avg in results.items():
        if avg is None:
            print(f"{model}: no valid data")
        else:
            print(f"{model}: {avg:.3f}")