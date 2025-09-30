import re

LANGS = {
    "en": "English",
    "de": "German",
    "it": "Italian"
}
GAMES = {
    "clean_up": "Clean Up",
    "dond": "Deal or No Deal",
    "hot_air_balloon": "Hot Air Balloon"
}

MODEL_PROPERTIES = {
    "claude-sonnet-4-20250514-no-reasoning-t1.0": {
        "Model Type": "commercial",
        "reasoning": "No",        
    }, 
    "claude-sonnet-4-20250514-t1.0": {
        "Model Type": "commercial",
        "reasoning": "Yes",                
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
    }
}

# model with reasoning traces 
MODEL_W_REASONING_TRACE = {
    "claude-sonnet-4-20250514-t1.0": {
        "Model Type": "commercial",
        "reasoning": "Yes",                
    },
    "claude-sonnet-4-20250514-t0.0": {
        "Model Type": "commercial",
        "reasoning": "Yes",
    }, 
    "deepseek-r1-distill-llama-70b-t1.0": {
        "Model Type": "open-weight",
        "reasoning": "Yes",
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
    }
}

ASSERT = 'ASSERT'
PROPOSE = 'PROPOSE'
UNDERMINE = 'UNDERMINE'
ALTERNATIVE = 'ALTERNATIVE'
CONCLUDE = 'CONCLUDE'

KEYWORDS = {
    "en": {
        ASSERT: ['need', 'should', 'must'],
        PROPOSE: ['maybe', 'perhaps', 'can', 'could'], 
        UNDERMINE: ['but', 'however', 'wait'], 
        ALTERNATIVE: ['alternatively', 'another'], 
        CONCLUDE: ['so', 'thus']
    }
}

TOKENIZER = re.compile(r"\b\w+\b")