import os
import json
import ast
from itertools import chain, combinations

def extract_summary(log_file: str, game_instance_file: str, output_file: str):
    with open(log_file, "r") as f:
        log_data = json.load(f)
    with open(game_instance_file, "r") as f:
        game_instance = json.load(f)

    item_weights = game_instance["item_weights"]
    p1_prefs = game_instance["player1_preferences"]
    p2_prefs = game_instance["player2_preferences"]

    pareto_sets = [frozenset(s) for s in game_instance.get("pareto_optima_sets", [])] \
                  if not game_instance.get("only_individual", False) else None

    def is_pareto(items):
        if pareto_sets is None:
            return None
        return frozenset(items) in pareto_sets

    def evaluate_set(items):
        total_weight = sum(item_weights.get(i, 0) for i in items)
        u1 = sum(p1_prefs.get(i, 0) for i in items)
        u2 = sum(p2_prefs.get(i, 0) for i in items)
        return total_weight, u1, u2

    def score_set(items):
        """Return full scoring dict for a proposal/agreement set."""
        total_weight, u1, u2 = evaluate_set(items)

        normalized_u1 = (u1 / game_instance["max_u1"] * 100) if game_instance["max_u1"] > 0 else float("NaN")
        normalized_u2 = (u2 / game_instance["max_u2"] * 100) if game_instance["max_u2"] > 0 else float("NaN")

        if normalized_u1 + normalized_u2 > 0:
            harmonic_mean = 2 * (normalized_u1 * normalized_u2) / (normalized_u1 + normalized_u2)
        else:
            harmonic_mean = float("NaN")

        normalized_harmonic_mean = (
            harmonic_mean / game_instance["max_harmonic"] * 100
            if game_instance["max_harmonic"] > 0 else float("NaN")
        )

        if game_instance["negotiation_mode"] == "equal":
            normalized_harmonic_mean = normalized_u1

        if total_weight > game_instance["max_weight"]:
            normalized_u1 = 0
            normalized_u2 = 0
            u1 = 0
            u2 = 0
            normalized_harmonic_mean = 0

        return {
            "items": sorted(list(items)),
            "weight": total_weight,
            "utility_player1": u1,
            "utility_player2": u2,
            "normalized_u1": normalized_u1,
            "normalized_u2": normalized_u2,
            "harmonic_mean": harmonic_mean,
            "normalized_harmonic_mean": normalized_harmonic_mean,
            "pareto_optimum": is_pareto(items),
        }

    # DP knapsack for individual optima
    def dp_pseudo_poly_knapsack(weights, values, max_weight):
        item_list = list(weights.keys())
        n = len(item_list)
        dp = [[0] * (max_weight + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            item = item_list[i - 1]
            w_i = weights[item]
            v_i = values[item]
            for w in range(max_weight + 1):
                if w_i > w:
                    dp[i][w] = dp[i-1][w]
                else:
                    dp[i][w] = max(dp[i-1][w], dp[i-1][w - w_i] + v_i)

        selected_items = set()
        w = max_weight
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                item = item_list[i-1]
                selected_items.add(item)
                w -= weights[item]

        max_value = dp[n][max_weight]
        return max_value, selected_items

    summary = {
        "results_folder": log_data["meta"]["results_folder"],
        "game_id": log_data["meta"]["game_id"],
        "proposals": [],
        "agreement": None,
        "max_weight": game_instance["max_weight"],
        "scores": {}
    }

    turns = 0
    final_deal = None

    for turn in log_data["turns"]:
        turns += 1
        for event in turn:
            if event["action"]["type"] == "get message":
                response = event["action"]["content"]
                actor = event["from"]

                if "PROPOSAL:" in response:
                    try:
                        set_str = response.split("PROPOSAL:", 1)[1].split("\n", 1)[0].strip()
                        proposal = ast.literal_eval(set_str)
                        scored = score_set(proposal)
                        scored["by"] = actor
                        scored["turn"] = turns
                        summary["proposals"].append(scored)
                    except Exception:
                        pass

                if "AGREE:" in response:
                    try:
                        set_str = response.split("AGREE:", 1)[1].split("\n", 1)[0].strip()
                        agreement = ast.literal_eval(set_str)
                        final_deal = agreement
                        scored = score_set(agreement)
                        scored["by"] = actor
                        scored["turn"] = turns
                        summary["agreement"] = scored
                    except Exception:
                        pass

    # Store maxima
    summary["max_u1"] = game_instance.get("max_u1")
    summary["max_u2"] = game_instance.get("max_u2")
    summary["max_harmonic"] = game_instance.get("max_harmonic")

    # Ensure proposals are sorted temporally
    summary["proposals"] = sorted(summary["proposals"], key=lambda x: x["turn"])

    # Compute final deal scores
    if final_deal:
        summary["scores"] = {
            "player1_score": summary["agreement"]["utility_player1"],
            "player2_score": summary["agreement"]["utility_player2"],
            "normalized_u1": summary["agreement"]["normalized_u1"],
            "normalized_u2": summary["agreement"]["normalized_u2"],
            "harmonic_mean": summary["agreement"]["harmonic_mean"],
            "normalized_harmonic_mean": summary["agreement"]["normalized_harmonic_mean"],
            "episode_score": 100.0 / turns if turns > 0 else 100.0,
            "pareto_optimum": summary["agreement"]["pareto_optimum"],
        }

    # Individual optima
    best_u1, set_u1 = dp_pseudo_poly_knapsack(item_weights, p1_prefs, game_instance["max_weight"])
    best_u2, set_u2 = dp_pseudo_poly_knapsack(item_weights, p2_prefs, game_instance["max_weight"])

    summary["player1_optimum"] = {
        "items": sorted(list(set_u1)),
        "utility_player1": best_u1,
        "utility_player2": sum(p2_prefs[i] for i in set_u1),
        "pareto_optimum": is_pareto(set_u1),
    }
    summary["player2_optimum"] = {
        "items": sorted(list(set_u2)),
        "utility_player1": sum(p1_prefs[i] for i in set_u2),
        "utility_player2": best_u2,
        "pareto_optimum": is_pareto(set_u2),
    }

    # Global optimum
    if not game_instance.get("only_individual", False):
        item_list = list(item_weights.keys())
        best_set = set()
        best_score = -1

        def all_subsets(iterable):
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

        for subset in all_subsets(item_list):
            total_weight, u1, u2 = evaluate_set(subset)
            if total_weight > game_instance["max_weight"]:
                continue

            norm_u1 = (u1 / game_instance["max_u1"] * 100) if game_instance["max_u1"] > 0 else 0
            norm_u2 = (u2 / game_instance["max_u2"] * 100) if game_instance["max_u2"] > 0 else 0

            harmonic = 2 * (norm_u1 * norm_u2) / (norm_u1 + norm_u2) if (norm_u1 + norm_u2) > 0 else 0

            if harmonic > best_score:
                best_score = harmonic
                best_set = set(subset)

        summary["global_optimum"] = {
            "items": sorted(list(best_set)),
            "utility_player1": sum(p1_prefs[i] for i in best_set),
            "utility_player2": sum(p2_prefs[i] for i in best_set),
            "harmonic_mean": best_score,
            "pareto_optimum": is_pareto(best_set),
        }
    else:
        summary["global_optimum"] = None  # not applicable

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def traverse_results(base_path: str):
    for model in os.listdir(base_path):
        model_path = os.path.join(base_path, model)
        if not os.path.isdir(model_path):
            continue

        hot_air_path = os.path.join(model_path, "hot_air_balloon")
        if not os.path.exists(hot_air_path):
            continue

        for experiment in os.listdir(hot_air_path):
            exp_path = os.path.join(hot_air_path, experiment)
            if not os.path.isdir(exp_path):
                continue

            for instance in os.listdir(exp_path):
                inst_path = os.path.join(exp_path, instance)
                if not os.path.isdir(inst_path):
                    continue

                log_file = os.path.join(inst_path, "interactions.json")
                game_instance_file = os.path.join(inst_path, "instance.json")
                output_file = os.path.join(inst_path, "summary.json")

                if os.path.exists(log_file) and os.path.exists(game_instance_file):
                    try:
                        extract_summary(log_file, game_instance_file, output_file)
                        print(f"Wrote summary for {inst_path}")
                    except Exception as e:
                        print(f"Failed at {inst_path}: {e}")


if __name__ == "__main__":
    # assumes to be run from root
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results_en"))
    traverse_results(base_path)
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results_de"))
    traverse_results(base_path)