import os
import json
import numpy as np

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

                        scores = summary_data.get("scores", {})

                        pareto_rate = scores.get("pareto_adherence_rate", None)
                        if pareto_rate is not None:
                            adherence_rates.append(pareto_rate)

                        alternation_rate = scores.get("alternation_rate", None)
                        if alternation_rate is not None:
                            alternation_rates.append(alternation_rate)

                    except Exception as e:
                        print(f"Failed to read {summary_file}: {e}")

        model_results[model] = {
            "avg_pareto_adherence_rate": float(np.mean(adherence_rates)) if adherence_rates else None,
            "avg_alternation_rate": float(np.mean(alternation_rates)) if alternation_rates else None,
        }

    return model_results


if __name__ == "__main__":
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results_en"))
    results = compute_model_metrics(base_path)

    print("\nAverage metrics per model:")
    for model, metrics in results.items():
        pareto = metrics["avg_pareto_adherence_rate"]
        alternation = metrics["avg_alternation_rate"]

        pareto_str = f"{pareto:.3f}" if pareto is not None else "no data"
        alternation_str = f"{alternation:.3f}" if alternation is not None else "no data"

        print(f"{model}: Pareto adherence = {pareto_str}, Alternation = {alternation_str}")