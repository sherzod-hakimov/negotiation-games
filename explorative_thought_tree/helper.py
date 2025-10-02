
TOKENIZER = re.compile(r"\b\w+\b")

# ⚙️: preprocessing
# 💯: 100% percent sure
# ✅: finished task but not 100% sure
# ⚠️: can't finish task, skip
def get_text_labels(text, lang='en', debug=True):
    """
    Label sentences based on keyword presence with conflict resolution.
    
    Args:
        text: Input string to process
    
    Returns:
        List of labels for sentences containing keywords
    """
    text = text.lower()
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    labels = []
    
    for i, sentence in enumerate(sentences):
        if debug: 
            print(f"------ processing sentence {i} ------\n{sentence} ")
        
        words = set(TOKENIZER.findall(sentence))
        
        # Find all matching labels for this sentence
        matching_labels = [
            label
            for label, keywords in KEYWORDS[lang].items()
            if words.intersection(keywords)
        ]
            
        if not matching_labels:
            continue  

        if CONCLUDE in matching_labels and len(matching_labels) > 1: 
            if debug: 
                print(f"⚙️  multiple labels containing CONCLUDE; remove CONCLUDE")
            matching_labels.remove(CONCLUDE)
        
        # Handle label conflicts and priority
        if len(matching_labels) == 1:
            if debug: 
                print(f"💯  selected label: {matching_labels}")
            labels.append(matching_labels[0])
        elif PROPOSE in matching_labels: 
            if debug: 
                print(f"✅  multiple labels containing PROPOSE; selected PROPOSE")
            labels.append(PROPOSE)
        else: 
            # Warn about non-PROPOSE conflicts
            if debug: 
                print(f"⚠️  Warning: conflicting labels: {matching_labels}; skip")
    
    return labels


    