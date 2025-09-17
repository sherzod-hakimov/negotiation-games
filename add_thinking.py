from copy import deepcopy
from math import exp
import langdetect
import json
import os
from typing import Dict

def find_experiment_dirs(base_dir: str, game_name: str = ""):
    experiment_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if "interactions.json" in files and "instance.json" in files and game_name in root:
            experiment_dirs.append(root)
    print(  f"found {len(experiment_dirs)} experiment dirs in {base_dir} with game_name '{game_name}'"  )
    return experiment_dirs

def think_event(player: str, thinking: str, languages: Dict[str, int]):
    thinking_event = {
        "from": player,
        "to": player,
        "action": {
            "type": "thinking",
            "content": thinking,
        }
    }
    if thinking and thinking.strip():
        try:
            language = langdetect.detect(thinking[:1000])  # only use the first 1000 characters for language detection
            if language not in languages:
                languages[language] = 0
            languages[language] += 1
            thinking_event["action"]["language"] = language
        except langdetect.lang_detect_exception.LangDetectException:
            print(f"could not detect language for thinking: {thinking[:100]}...")
            pass
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
    new_interactions = deepcopy(interactions)
    new_interactions["turns"] = []
    think_languages = {}
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
                                thinking_event = think_event(event["from"], thinking, think_languages)
                                new_interactions["turns"][-1].append(thinking_event)
                        elif "choices" in raw_response_object:
                            message = raw_response_object["choices"][0]["message"]
                            if "reasoning" in message:
                                thinking = message["reasoning"]
                                thinking_event = think_event(event["from"], thinking, think_languages)
                                if "error" in raw_response_object["choices"][0]:
                                    thinking_event["action"]["content"] += f"\nERROR: {raw_response_object['choices'][0]['error']['message']}"
                                new_interactions["turns"][-1].append(thinking_event)
            new_interactions["turns"][-1].append(event)
    if think_languages:
        new_interactions["think_languages"] = think_languages
    if interactions != new_interactions:
        with open(os.path.join(experiment_dir, "interactions_with_thinking.json"), "w") as f:
            json.dump(new_interactions, f, indent=4)
    return think_languages

if __name__ == "__main__":
    epxeriment_roots = ["/Users/karlosswald/repositories/clemclass/negotiation-games/results_en", "/Users/karlosswald/repositories/clemclass/negotiation-games/results_de"]
    # epxeriment_roots = ["/Users/karlosswald/repositories/clemclass/negotiation-games/results_de"]
    for base_dir in epxeriment_roots:
        language_dict = {}
        experiment_dirs = find_experiment_dirs(base_dir) #, game_name="clean_up")
        for experiment_dir in experiment_dirs:
            think_languages = thinking_to_interactions(experiment_dir)
            if len(think_languages) > 1:
                model = experiment_dir.split("/")[-4]
                game = experiment_dir.split("/")[-3]
                experiment = experiment_dir.split("/")[-2]
                instance = experiment_dir.split("_")[-1]
                if model not in language_dict:
                    language_dict[model] = {}
                if game not in language_dict[model]:
                    language_dict[model][game] = {}
                if experiment not in language_dict[model][game]:
                    language_dict[model][game][experiment] = {}
                language_dict[model][game][experiment][instance] = think_languages
        with open(os.path.join(base_dir, "thinking_languages.json"), "w") as f:
            json.dump(language_dict, f, indent=4)
            print(f"wrote thinking languages to {os.path.join(base_dir, 'thinking_languages.json')}")