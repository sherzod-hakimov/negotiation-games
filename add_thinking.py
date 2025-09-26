from copy import deepcopy
from email.mime import base
import json
import os

def find_experiment_dirs(base_dir: str, game_name: str = ""):
    experiment_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if "interactions.json" in files and "instance.json" in files and game_name in root:
            experiment_dirs.append(root)
    print(  f"found {len(experiment_dirs)} experiment dirs in {base_dir} with game_name '{game_name}'"  )
    return experiment_dirs

def think_event(player: str, thinking: str, timestamp: str = ""):
    if not thinking:
        thinking = ""
    thinking_event = {
        "from": player,
        "to": player,
        "timestamp": timestamp,
        "action": {
            "type": "thinking",
            "content": thinking,
        }
    }
    return thinking_event

def thinking_to_interactions(experiment_dir: str):
    interaction_file = os.path.join(experiment_dir, "interactions.json")
    requests_file = os.path.join(experiment_dir, "requests.json")
    # check if files exist
    if not os.path.exists(interaction_file) or not os.path.exists(requests_file):
        print(f"skipping {experiment_dir} because interaction or requests file does not exist")
        return {}
    with open(interaction_file, "r") as f:
        interactions = json.load(f)
    with open(requests_file, "r") as f:
        requests = json.load(f)
    thinking_found = False
    new_interactions = deepcopy(interactions)
    new_interactions["turns"] = []
    for turn in interactions["turns"]:
        new_interactions["turns"].append([])
        for event in turn:
            if event["from"].startswith("Player") and event["to"] == "GM":
                timestamp = event.get("timestamp", "")
                for request in requests:
                    if timestamp == request["timestamp"]:
                        raw_response_object = request["raw_response_obj"]
                        if "content" in raw_response_object and len(raw_response_object["content"]) == 2:
                            raw_response_content = raw_response_object["content"]
                            if "thinking" in raw_response_content[0]:
                                thinking_found = True
                                thinking = raw_response_content[0]["thinking"]
                                thinking_event = think_event(event["from"], thinking, timestamp)
                                new_interactions["turns"][-1].append(thinking_event)
                        elif "choices" in raw_response_object:
                            message = raw_response_object["choices"][0]["message"]
                            if "reasoning" in message:
                                thinking_found = True
                                thinking = message["reasoning"]
                                thinking_event = think_event(event["from"], thinking, timestamp)
                                if "error" in raw_response_object["choices"][0]:
                                    thinking_event["action"]["content"] += f"\nERROR: {raw_response_object['choices'][0]['error']['message']}"
                                new_interactions["turns"][-1].append(thinking_event)
            new_interactions["turns"][-1].append(event)

    if thinking_found:
        with open(os.path.join(experiment_dir, "interactions_with_thinking.json"), "w") as f:
            json.dump(new_interactions, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    epxeriment_roots = [
        "/Users/karlosswald/repositories/clemclass/negotiation-games/results_en", 
        "/Users/karlosswald/repositories/clemclass/negotiation-games/results_de",
        "/Users/karlosswald/repositories/clemclass/negotiation-games/results_it"
    ]
    # epxeriment_roots = ["/Users/karlosswald/repositories/clemclass/negotiation-games/results_de"]
    for base_dir in epxeriment_roots:
        experiment_dirs = find_experiment_dirs(base_dir) #, game_name="clean_up")
        for experiment_dir in experiment_dirs:
            thinking_to_interactions(experiment_dir)