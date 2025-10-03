from lingua import LanguageDetectorBuilder
import os
import requests
import re
import json

def find_experiment_dirs(base_dir: str, game_name: str = ""):
    experiment_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if "interactions.json" in files and "instance.json" in files and game_name in root:
            experiment_dirs.append(root)
    print(  f"found {len(experiment_dirs)} experiment dirs in {base_dir} with game_name '{game_name}'"  )
    return experiment_dirs

LANGUAGE_STATS_DIR = "./language_stats"

LANGUAGE_DETECTOR = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().with_minimum_relative_distance(.4).build()
# LANGUAGE_DETECTOR = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()

def split_experiment_dir(experiment_dir: str):
    game_language = experiment_dir.split("/")[-5].split("_")[-1]
    model = experiment_dir.split("/")[-4]
    game = experiment_dir.split("/")[-3]
    experiment = experiment_dir.split("/")[-2]
    instance = experiment_dir.split("_")[-1]
    return game_language, model, game, experiment, instance

def load_unicode_blocks(block_filename):
    if not os.path.exists(block_filename):
        print('Unicode block file %s does not exist. Downloading…' % block_filename)
        r = requests.get('http://unicode.org/Public/UNIDATA/Blocks.txt')
        r.raise_for_status()

        with open(block_filename, 'wb') as f:
            for chunk in r.iter_content():
                f.write(chunk)

    with open(block_filename, 'rb') as f:
        blocks = load_unicode_blocks_from_file(f)

    return blocks


def load_unicode_blocks_from_file(f):
    file_contents = f.read().decode('utf-8')

    blocks = []
    for start, end, block_name in re.findall(r'([0-9A-F]+)\.\.([0-9A-F]+);\ (\S.*\S)', file_contents):
        if block_name == 'No_Block':
            continue

        blocks.append((int(start, 16), int(end, 16), block_name))

    return blocks

UNICODE_BLOCKS = load_unicode_blocks('UNIDATA-Blocks.txt')

# merge some blocks into broader categories
merge_blocks =[
    ("Latin", "Latin"),
    ("Cyrillic", "Cyrillic"),
    ("CJK", "CJK"),
    ("Katakana", "CJK"),
    ("Hiragana", "CJK"),
    ("Arabic", "Arabic"),
    ("Greek and Coptic", "Greek"),
    ("Hangul Syllables", "Hangul"),
    ("Hangul Jamo", "Hangul"),
]


def load_merged_blocks():
    merged = []
    for start, end, block_name in UNICODE_BLOCKS:
        merged_name = None
        for search, replace in merge_blocks:
            if search in block_name:
                merged_name = replace
                break
        # if 'Extended' or 'Supplement' in block_name: merge into base block
        if 'Extended' in block_name:
            block_name = block_name.split(' Extended')[0]
        if 'Supplement' in block_name:
            block_name = block_name.split(' Supplement')[0]

        if merged_name is None:
            merged_name = block_name
        merged.append((start, end, merged_name, block_name))
    return merged

MERGED_BLOCKS = load_merged_blocks()

IGNORE_BLOCKS = [
    'Box Drawing',
    'Block Elements',
    'Geometric Shapes',
    'Miscellaneous Symbols and Arrows',
    'General Punctuation',
    'Arrows',
    'Dingbats',
    'Supplemental Punctuation',
    'Mathematical Operators',
    'Variation Selectors',
    'Miscellaneous Mathematical Symbols-A',
    'Miscellaneous Mathematical Symbols-B',
    'Specials',
    'Currency Symbols',
    'Halfwidth and Fullwidth Forms',
    'Miscellaneous Technical',
    'Miscellaneous Symbols',
    'Number Forms',
    'Enclosed Alphanumerics',
    'Letterlike Symbols',
    'Optical Character Recognition',
    'Miscellaneous Symbols and Pictographs',
    'Emoticons',
    'Control Pictures',
    'Transport and Map Symbols',
    'Superscripts and Subscripts',
    'Braille Patterns',
    'Spacing Modifier Letters',
    'Combining Diacritical Marks',
    'Mathematical Alphanumeric Symbols',
    'Supplemental Arrows-A',
    'Supplemental Arrows-B',
    'Supplemental Arrows-C',
    'Hangul Compatibility Jamo',
    'Private Use Area',
    'Phonetic Extensions',
]

def script_for_character(char):
    # escape whitespace, control characters, etc.
    escape_chars = " \t\n.,:;!?()[]{}<>\"'`~@#$%^&*-+=|\\/"
    if char in escape_chars:
        return None
    codepoint = ord(char)
    for start, end, type, block_name in MERGED_BLOCKS:
        if start <= codepoint <= end:
            # ignore certain blocks
            if block_name in IGNORE_BLOCKS:
                return None
            return type
    return None

def get_script_stats(text):
    if not text:
        return None

    scripts = {}
    current_block = script_for_character(text[0])
    start_index = 0

    for i, char in enumerate(text):
        block = script_for_character(char)
        if block is None:
            block = current_block
        else:
            if block not in scripts:
                scripts[block] = {'count': 0, 'segments': []}
            scripts[block]['count'] += 1
            if block != current_block and current_block is not None:
                scripts[current_block]['segments'].append((start_index, i))
                current_block = block
                start_index = i
    return scripts

def get_main_language(text):
    # if len(text) < 100:
    #     return "UNK"
    # orig_text = text
    # patterns_to_remove = [
    #     r"AGREE:\s*\{.*?\}",
    #     r"REFUSE:\s*\{.*?\}",
    #     r"PROPOSAL:\s*\{.*?\}",
    #     r"ZUSTIMMUNG:\s*\{.*?\}",
    #     r"ABLEHNUNG:\s*\{.*?\}",
    #     r"VORSCHLAG:\s*\{.*?\}",
    #     r"PROPOSTA:\s*\{.*?\}",
    #     r"ACCORDO:\s*\{.*?\}",
    #     r"RIFIUTO:\s*\{.*?\}",
    #     r"SPOSTA:\s*\w,\s*\(\d+,\s*\d+\)",
    #     r"MOVE:\s*\w,\s*\(\d+,\s*\d+\)",
    #     r"BEWEGE:\s*\w,\s*\(\d+,\s*\d+\)",
    #     r"\w,?\s*\(\d+,\s*\d+\)",
    #     r"\{.*?\}"
    # ]
    # # remove the above patterns from text
    # for pattern in patterns_to_remove:
    #     text = re.sub(pattern, "", text, flags=re.DOTALL)
    
    # if len(text) < 50:
    #     return "UNK"
    language = LANGUAGE_DETECTOR.detect_language_of(text)
    if language is None:
        return "UNK"
    return str(language.iso_code_639_1).split('.')[1].lower()

def detect_thinking_languages(experiment_dir: str):
    if 'interactions_with_thinking.json' not in os.listdir(experiment_dir):
        interaction_file = os.path.join(experiment_dir, "interactions.json")
        return
    interaction_file = os.path.join(experiment_dir, "interactions_with_thinking.json")
    game_language, model, game, experiment, instance = split_experiment_dir(experiment_dir)
    with open(interaction_file, "r") as f:
        interactions = json.load(f)
    check_events = []
    for turn in interactions["turns"]:
        for event in turn:
            if event["action"]["type"] in ["thinking", "get message"]:
                check_events.append(event)
            # if event["action"]["type"] == "get message":
            #     main_language = get_main_language(event["action"]["content"])
            #     if main_language != game_language:
            #         print(f"Message language {main_language} does not match game language {game_language} in {interaction_file}")
            #         print(event["action"]["content"])
            #         print("-----")
            #     scripts = get_script_stats(event["action"]["content"])
            #     if scripts and len(scripts) > 1:
            #         print(f"Message has multiple scripts {list(scripts.keys())} in {interaction_file}")
            #         print(event["action"]["content"])
            #         print("-----")
    if not check_events:
        return
    script_excerpts = []
    script_counts = {}
    language_excerpts = []
    language_counts = {}
    for event in check_events:
        type = "response"
        if event["action"]["type"] == "thinking":
            type = "thinking"
        text = event["action"]["content"]
        if not text or text.strip() == "":
            continue
        scripts = get_script_stats(text)
        main_language = get_main_language(text)
        if main_language == "UNK":
            continue
        if type not in language_counts:
            language_counts[type] = {}
        if main_language not in language_counts[type]:
            language_counts[type][main_language] = 0
        language_counts[type][main_language] += 1
        if main_language not in ["en", "de", "it", "UNK"]:
            language_excerpts.append({
                'type': type,
                'language': main_language,
                'length': len(text),
                'text': text,
                'timestamp': event.get('timestamp', '')
            })
        if scripts:
            for script, stats in scripts.items():
                if type not in script_counts:
                    script_counts[type] = {}
                if script not in script_counts:
                    script_counts[type][script] = 0
                script_counts[type][script] += stats['count']
                if script == 'Latin':
                    continue
                for segment in stats['segments']:
                    seg_range = 100
                    start_idx = max(0, segment[0]-seg_range)
                    end_idx = min(len(text), segment[1]+seg_range)
                    segment_text = text[start_idx:end_idx]
                    script_excerpts.append({
                        'type': type,
                        'script': script,
                        'length': segment[1]-segment[0],
                        'text': text[segment[0]:segment[1]],
                        'context': segment_text,
                        'start': segment[0],
                        'end': segment[1],
                        'timestamp': event.get('timestamp', '')
                    })

    if language_excerpts:
        excerpts_file = os.path.join(LANGUAGE_STATS_DIR, "language_excerpts.json")
        excerpts = {}
        if os.path.exists(excerpts_file):
            with open(excerpts_file, "r") as f:
                excerpts = json.load(f)
        for excerpt in language_excerpts:
            if model not in excerpts:
                excerpts[model] = {}
            if game_language not in excerpts[model]:
                excerpts[model][game_language] = {}
            if game not in excerpts[model][game_language]:
                excerpts[model][game_language][game] = {}
            if experiment not in excerpts[model][game_language][game]:
                excerpts[model][game_language][game][experiment] = {}
            if instance not in excerpts[model][game_language][game][experiment]:
                excerpts[model][game_language][game][experiment][instance] = []
            excerpts[model][game_language][game][experiment][instance].append(excerpt)
        with open(excerpts_file, "w") as f:
            json.dump(excerpts, f, indent=4, ensure_ascii=False)
        # print(f"wrote {len(language_excerpts)} language excerpts to {excerpts_file}")

    for type in language_counts:
        for language in language_counts[type]:
            csv_file = os.path.join(LANGUAGE_STATS_DIR, "language_stats.csv")
            # check if file exists
            if not os.path.exists(csv_file):
                with open(csv_file, "w") as f:
                    f.write("game_language,game,model,experiment,instance,type,language,count\n")
            with open(csv_file, "a") as f:
                f.write(f"{game_language},{game},{model},{experiment},{instance},{type},{language},{language_counts[type][language]}\n")
    # for language in language_counts:
    #     csv_file = os.path.join(LANGUAGE_STATS_DIR, "language_stats.csv")
    #     # check if file exists
    #     if not os.path.exists(csv_file):
    #         with open(csv_file, "w") as f:
    #             f.write("game_language,game,model,experiment,instance,type,language,count\n")
    #     with open(csv_file, "a") as f:
    #         f.write(f"{game_language},{game},{model},{experiment},{instance},{type},{language},{language_counts[language]}\n")
    if script_excerpts:
        excerpts_file = os.path.join(LANGUAGE_STATS_DIR, "script_excerpts.json")
        excerpts = {}
        if os.path.exists(excerpts_file):
            with open(excerpts_file, "r") as f:
                excerpts = json.load(f)
        for excerpt in script_excerpts:
            if model not in excerpts:
                excerpts[model] = {}
            if game_language not in excerpts[model]:
                excerpts[model][game_language] = {}
            if game not in excerpts[model][game_language]:
                excerpts[model][game_language][game] = {}
            if experiment not in excerpts[model][game_language][game]:
                excerpts[model][game_language][game][experiment] = {}
            if instance not in excerpts[model][game_language][game][experiment]:
                excerpts[model][game_language][game][experiment][instance] = []
            excerpts[model][game_language][game][experiment][instance].append(excerpt)
        with open(excerpts_file, "w") as f:
            json.dump(excerpts, f, indent=4, ensure_ascii=False)
        # print(f"wrote {len(script_excerpts)} excerpts to {excerpts_file}")

    for type in script_counts:
        for script, count in script_counts[type].items():
            csv_file = os.path.join(LANGUAGE_STATS_DIR, "script_stats.csv")
            # check if file exists
            if not os.path.exists(csv_file):
                with open(csv_file, "w") as f:
                    f.write("game_language,game,model,experiment,instance,type,script,count\n")
            with open(csv_file, "a") as f:
                f.write(f"{game_language},{game},{model},{experiment},{instance},{type},{script},{count}\n")
    # for script, count in script_counts.items():
    #     csv_file = os.path.join(LANGUAGE_STATS_DIR, "script_stats.csv")
    #     # check if file exists
    #     if not os.path.exists(csv_file):
    #         with open(csv_file, "w") as f:
    #             f.write("game_language,game,model,experiment,instance,script,count\n")
    #     with open(csv_file, "a") as f:
    #         f.write(f"{game_language},{game},{model},{experiment},{instance},{script},{count}\n")

if __name__ == "__main__":
    epxeriment_roots = [
        "/Users/karlosswald/repositories/clemclass/negotiation-games/results_en", 
        "/Users/karlosswald/repositories/clemclass/negotiation-games/results_de",
        "/Users/karlosswald/repositories/clemclass/negotiation-games/results_it"
    ]
    # remove old script_excerpts and script_stats, if exists
    for fname in ["script_excerpts.json", "script_stats.csv", "language_excerpts.json", "language_stats.csv"]:
        fpath = os.path.join(LANGUAGE_STATS_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
    for base_dir in epxeriment_roots:
        experiment_dirs = find_experiment_dirs(base_dir) #, game_name="clean_up")
        for experiment_dir in experiment_dirs:
            detect_thinking_languages(experiment_dir)
