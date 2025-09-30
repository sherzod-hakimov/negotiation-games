# import logging
import os 
import csv
import gc
import json

import pandas as pd
from pathlib import Path


from constant import MODEL_PROPERTIES, MODEL_W_REASONING_TRACE, TOKENIZER, KEYWORDS
from constant import ASSERT, PROPOSE, UNDERMINE, ALTERNATIVE, CONCLUDE


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='debug.log',
    filemode='a'  
)

def transform(df, metrics_to_keep=None): 
    """
    Take a raw.csv, add columns of model properties, only keep rows of certain metrics, 
    and pivot these metrics as columns. 
    """
    # add / drop columns
    df[["Model Type", "reasoning"]] = df["model"].map(MODEL_PROPERTIES).apply(pd.Series)
    df = df.drop(['Unnamed: 0'], axis=1)    

    # drop rows
    if not metrics_to_keep: 
        metrics_to_keep = ['Main Score', 
                            'Success', 'Aborted']
    df = df.loc[df['metric'].isin(metrics_to_keep)]

    df_pv = df.pivot_table(
            index=["game", "model", "experiment", "episode", "Model Type", "reasoning"],
            columns="metric",
            values="value",
            aggfunc="first" # no actual aggregation will happen 
        ).reset_index()

    df_pv = df_pv.astype(
        { 'Main Score': 'float',  
          'Success': 'float', 
          'Aborted': 'float'
        })
    
    df_pv = df_pv.astype(
        {'Success': 'int', 
         'Aborted': 'int'})    

    return df_pv


# ⚙️: preprocessing
# 💯: 100% percent sure
# ✅: finished task but not 100% sure
# ⚠️: can't finish task, skip
def get_text_labels(text, lang='en', debug=True):
    """
    Label sentences based on keyword presence with simple conflict resolution rules.
    
    Args:
        text: Input string to process, it's the content of a thinking event. 
    
    Returns:
        List of labels for sentences containing keywords
    """
    text = text.lower()
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    labels = []
    
    for i, sentence in enumerate(sentences):
        logging.debug(f"------ processing sentence {i} ------\n{sentence} ")

        words = set(TOKENIZER.findall(sentence))
        
        # Find all matching labels for this sentence
        matching_labels = [
            label
            for label, keywords in KEYWORDS[lang].items()
            if words.intersection(keywords)
        ]
            
        if not matching_labels:
            continue  

        # -- Handle label conflicts and priority --
        # for sentences like "so perhaps ..." it's a PROPOSAL  
        if CONCLUDE in matching_labels and len(matching_labels) > 1: 
            logging.debug(f"⚙️  multiple labels containing CONCLUDE; remove CONCLUDE")
            matching_labels.remove(CONCLUDE)
        
        if len(matching_labels) == 1:
            logging.debug(f"💯  selected label: {matching_labels}")
            labels.append(matching_labels[0])
        # sentences like "so perhaps we should ...", PROPOSAL dwarfs the other labels
        elif PROPOSE in matching_labels: 
            logging.debug(f"✅  multiple labels containing PROPOSE; selected PROPOSE")
            labels.append(PROPOSE)
        else: 
            # Warn about non-PROPOSE conflicts
            logging.debug(f"⚠️  Warning: conflicting labels: {matching_labels}; skip")
    
    return labels    

def extract_reasoning(data): 
    # extract the 1st turn only
    extracted = []
    for obj in data['turns'][0]: 
        if obj['action']['type'] == "thinking": 
            extracted.append(obj)

    return extracted

def write_reasoning_csv(output_csv="reasoning_labels_per_episode.csv"):
    """
    Extract reasoning traces from the 1st turn of interactions, 
    transfrom them into `reasoning_raw` (thinking events as it is) 
    and `reasoning_labels` (see `get_text_label`), finally save them 
    together with lang, model, game, experiment, instance info in a csv. 
    """
    if os.path.exists(output_csv):
        os.remove(output_csv)
    
    for lang in KEYWORDS: 
        for model in MODEL_W_REASONING_TRACE:     
            root = Path(f"../results_{lang}/{model}")
            # <LANG>/<MODEL>/<GAME>/<EXPERIMENT>/<INSTANCE>/interactions_with_thinking.json
            json_files = list(root.glob("*/*/*/interactions_with_thinking.json"))
            total = len(json_files)
            
            file_exists = os.path.isfile(output_csv)
            
            with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
            
                # write header only if file did not exist before
                if not file_exists:
                    writer.writerow([
                        "lang", "model", "game", "experiment", "episode",
                        "reasoning_raw", "reasoning_labels"
                    ])
        
                for i, json_file in enumerate(json_files):
                    episode = json_file.parent.name
                    experiment = json_file.parent.parent.name
                    game = json_file.parent.parent.parent.name
        
                    print(f"[{lang}-{model} {i+1}/{total}] Processing: {game}/{experiment}/{episode}")
                    
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        
                        reasoning_raw_r = extract_reasoning(data)
                        reasoning_raw = json.dumps(reasoning_raw_r)
                        
                        logging.debug(f"====== Processing {lang}-{model}-{json_file} ======")
                        reasoning_labels_r = [
                            get_text_labels(e['action']['content'], lang=lang, debug=False) 
                            for e in reasoning_raw_r
                        ]
                        reasoning_labels = json.dumps(reasoning_labels_r)
                        
                        writer.writerow([lang, model, game, experiment, episode, 
                                       reasoning_raw, reasoning_labels])
                        
                        # Flush to disk periodically
                        if (i + 1) % 10 == 0:
                            csvfile.flush()
                        
                        # Clean up
                        del data, reasoning_raw_r, reasoning_raw, reasoning_labels_r, reasoning_labels
                        
                        # Garbage collect periodically
                        if (i + 1) % 20 == 0:
                            gc.collect()
                            
                    except KeyboardInterrupt:
                        print(f"\n⚠️  Interrupted at {i}/{total}")
                        break
                    except Exception as e:
                        print(f"  ❌ Error: {e}")
                        # Write error row with empty data
                        writer.writerow([lang, model, game, experiment, episode, 
                                       json.dumps({"error": str(e)}), "[]"])
                        continue
            
            print(f"\n✓ Completed: {output_csv}")
            gc.collect()    