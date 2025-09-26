import pandas as pd

MODEL_PROPERTIES = {
    "claude-sonnet-4-20250514-no-reasoning-t1.0": {  # what does the t1.0 mean here? 
        "Model Type": "commercial",
        "reasoning": "No",        
    }, 
    "claude-sonnet-4-20250514-t1.0": {
        "Model Type": "commercial",
        "reasoning": "No",                
    },
    "claude-sonnet-4-20250514-no-reasoning-t0.0": {
        "Model Type": "commercial",
        "reasoning": "No",
    }, 
    "claude-sonnet-4-20250514-t0.0": {
        "Model Type": "commercial",
        "reasoning": "Yes",
    }, 
    "gpt-5-2025-08-07-no-reasoning-t1.0": {
        "Model Type": "commercial",
        "reasoning": "No",
    },
    "gpt-5-2025-08-07-t1.0": {
        "Model Type": "commercial",
        "reasoning": "Yes",
    },
    "gpt-5-mini-2025-08-07-no-reasoning-t1.0": {
        "Model Type": "commercial",
        "reasoning": "No",
    },
    "gpt-5-mini-2025-08-07-t1.0": {
        "Model Type": "commercial",
        "reasoning": "Yes",
    },
    "llama-3.3-70b-instruct-t1.0": {
        "Model Type": "open-weight",
        "reasoning": "No",
    },
    "deepseek-r1-distill-llama-70b-t1.0": {
        "Model Type": "open-weight",
        "reasoning": "Yes",
    },
    "nemotron-nano-9b-v2-no-reasoning-t1.0": {
        "Model Type": "open-weight",
        "reasoning": "No",
    },
    "nemotron-nano-9b-v2-t1.0": {
        "Model Type": "open-weight",
        "reasoning": "Yes",
    }, 
    "gpt-oss-120b-t1.0": {
        "Model Type": "open-weight", 
        "reasoning": "Yes"
    }, 
    "qwen3-next-80b-a3b-thinking-t1.0": {
        "Model Type": "open-weight", 
        "reasoning": "Yes"
    }, 
    "qwen3-next-80b-a3b-instruct-t1.0": {
        "Model Type": "open-weight", 
        "reasoning": "No"
    }, 
    "deepseek-chat-v3.1-t1.0": {
        "Model Type": "open-weight", 
        "reasoning": "No"
    }, 
    
}


def transform(df): 
    # add / drop columns
    df[["Model Type", "reasoning"]] = df["model"].map(MODEL_PROPERTIES).apply(pd.Series)
    df = df.drop(['Unnamed: 0'], axis=1)    

    # drop rows
    to_keep = ['Main Score', 
            'Success', 'Aborted'
          ]
    df = df.loc[df['metric'].isin(to_keep)]

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