from copy import deepcopy
import json
import os

def find_experiment_dirs(base_dir: str, game_name: str = ""):
    experiment_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if "interactions.json" in files and "instance.json" in files and game_name in root:
            experiment_dirs.append(root)
    print(  f"found {len(experiment_dirs)} experiment dirs in {base_dir} with game_name '{game_name}'"  )
    return experiment_dirs

def thinking_to_interactions(experiment_dir: str):
    interaction_file = os.path.join(experiment_dir, "interactions.json")
    requests_file = os.path.join(experiment_dir, "requests.json")
    with open(interaction_file, "r") as f:
        interactions = json.load(f)
    with open(requests_file, "r") as f:
        requests = json.load(f)
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
                                thinking = raw_response_content[0]["thinking"]
                                thinking_event = {
                                    "from": event["from"],
                                    "to": event["from"],
                                    "action": {
                                        "type": "thinking",
                                        "content": thinking
                                    }
                                }
                                new_interactions["turns"][-1].append(thinking_event)
                        elif "choices" in raw_response_object:
                            message = raw_response_object["choices"][0]["message"]
                            if "reasoning" in message:
                                thinking = message["reasoning"]
                                thinking_event = {
                                    "from": event["from"],
                                    "to": event["from"],
                                    "action": {
                                        "type": "thinking",
                                        "content": thinking
                                    }
                                }
                                if "error" in raw_response_object["choices"][0]:
                                    thinking_event["action"]["content"] += f"\nERROR: {raw_response_object['choices'][0]['error']['message']}"
                                new_interactions["turns"][-1].append(thinking_event)
            new_interactions["turns"][-1].append(event)
    if interactions != new_interactions:
        with open(os.path.join(experiment_dir, "interactions_with_thinking.json"), "w") as f:
            json.dump(new_interactions, f, indent=4)
        print(f"saved interactions with thinking to {experiment_dir}/interactions_with_thinking.json")

if __name__ == "__main__":
    experiment_dirs = find_experiment_dirs("/Users/karlosswald/repositories/clemclass/negotiation-games/results_en", game_name="clean_up")
    experiment_dirs = experiment_dirs + find_experiment_dirs("/Users/karlosswald/repositories/clemclass/negotiation-games/results_de", game_name="clean_up")
    for experiment_dir in experiment_dirs:
        print(experiment_dir)
        thinking_to_interactions(experiment_dir)