import json


def translate_json_content_fully(input_filepath='instances_it.json', output_filepath='instances_it_translated.json'):
    """
    Reads a JSON file, translates specified English text fields, tags, and
    regex patterns to Italian, and saves the result to a new JSON file.

    Args:
        input_filepath (str): The path to the input JSON file.
        output_filepath (str): The path where the translated JSON file will be saved.
    """
    # Dictionary now includes tags with and without colons for comprehensive replacement.
    translations = {


        # Error Prompts
        "You refused a proposal which is not active. Proposals are only active if they have been logged by the other player via PROPOSAL and have not been deactivated by you via REFUSE. Try again.":
            "Hai rifiutato una proposta non attiva. Le proposte sono attive solo se sono state registrate dall'altro giocatore tramite PROPOSTA e non sono state disattivate da te tramite RIFIUTO. Riprova.",
        "You made more than one agreement. Final deals cannot be ambiguous. Try again":
            "Hai fatto più di un accordo. Gli accordi finali non possono essere ambigui. Riprova.",
        "You agreed to a proposal which is not active. Proposals are only active if they have been logged by the other player via PROPOSAL and have not been deactivated by you via REFUSE. Try again.":
            "Hai accettato una proposta non attiva. Le proposte sono attive solo se sono state registrate dall'altro giocatore tramite PROPOSTA e non sono state disattivate da te tramite RIFIUTO. Riprova.",
        "Your proposal includes items which are not in the game. Try again":
            "La tua proposta include oggetti che non sono nel gioco. Riprova.",
        "You used a PROPOSAL tag, but did not provide a valid python set containing strings as arguments, e.g. {'A', 'B', 'C', ...}. Try again.":
            "Hai usato un tag PROPOSTA, ma non hai fornito un set python valido contenente stringhe come argomenti, ad es. {'A', 'B', 'C', ...}. Riprova.",
        "Your response did not start with the proper strategic reasoning tag at the very beginning of your response. The very first structured format must be of the form STRATEGIC REASONING: {...}. Try again.":
            "La tua risposta non è iniziata con il tag di ragionamento strategico corretto all'inizio della tua risposta. Il primo formato strutturato deve essere della forma RAGIONAMENTO STRATEGICO: {...}. Riprova.",
        "Your response did not contain an argument. You must include the structured format ARGUMENT: {...} somewhere in your response. Try again":
            "La tua risposta non conteneva un argomento. Devi includere il formato strutturato ARGOMENTO: {...} da qualche parte nella tua risposta. Riprova.",
        "Your response contained an untagged sequence or you used STRATEGIC REASONING more than once. You may only use the structured formats as explained in the initial message. They must all be of the form TAG: {...}":
            "La tua risposta conteneva una sequenza non taggata o hai usato RAGIONAMENTO STRATEGICO più di una volta. Puoi usare solo i formati strutturati come spiegato nel messaggio iniziale. Devono essere tutti della forma TAG: {...}",
        "Your response only contained a strategic reasoning tag. You must at least include one more valid tag in your response, so that the other player receives a message. Try again.":
            "La tua risposta conteneva solo un tag di ragionamento strategico. Devi includere almeno un altro tag valido nella tua risposta, in modo che l'altro giocatore riceva un messaggio. Riprova.",

        # Initial Prompts - Full sentences and phrases
        "You are participating in a collaborative negotiation game.":
            "Stai partecipando a un gioco di negoziazione collaborativa.",
        "Together with another participant, you must agree on a single set of items that will be kept. Each of you has your own view of how much each item matters to you (importance). You do not know how the other participant values the items. Additionally, you are given the effort each item demands.":
            "Insieme a un altro partecipante, dovete accordarvi su un unico set di oggetti da conservare. Ognuno di voi ha la propria visione di quanto ogni oggetto sia importante (importanza). Non sai come l'altro partecipante valuta gli oggetti. Inoltre, ti viene fornito lo sforzo che ogni oggetto richiede.",
        "You may only agree on a set if the total effort of the selected items does not exceed a shared limit:":
            "Potete accordarvi su un set solo se lo sforzo totale degli oggetti selezionati non supera un limite condiviso:",
        "Here are the individual item effort values:": "Ecco i valori di sforzo dei singoli oggetti:",
        "Here is your personal view on the importance of each item:": "Ecco la tua visione personale sull'importanza di ogni oggetto:",
        "Your goal is to negotiate a shared set of items that benefits you as much as possible (i.e., maximizes total importance to YOU), while staying within the LIMIT. You are not required to make a PROPOSAL in every message - you can simply negotiate as well. All tactics are allowed!":
            "Il tuo obiettivo è negoziare un set condiviso di oggetti che ti avvantaggi il più possibile (cioè, massimizzi l'importanza totale per TE), rimanendo entro il LIMITE. Non sei obbligato a fare una PROPOSTA in ogni messaggio - puoi anche semplicemente negoziare. Tutte le tattiche sono permesse!",
        "You may only use the following structured formats in a message:": "Puoi usare solo i seguenti formati strutturati in un messaggio:",
        "Propose keeping exactly those items.": "Proponi di conservare esattamente questi oggetti.",
        "Explicitly reject opponent's proposal.": "Rifiuta esplicitamente la proposta dell'avversario.",
        "Defend your last proposal or argue against the player's proposal.": "Difendi la tua ultima proposta o argomenta contro la proposta del giocatore.",
        "Accept the opponent's proposal which ends the game.": "Accetta la proposta dell'avversario che termina il gioco.",
        "Describe your strategic reasoning or anticipation explaining your choice of action. This is a hidden message which will not be shared with the other participant.":
            "Descrivi il tuo ragionamento strategico o anticipazione spiegando la tua scelta di azione. Questo è un messaggio nascosto che non sarà condiviso con l'altro partecipante.",
        "You may only AGREE on a proposal the other party has logged via PROPOSAL.": "Puoi ACCETTARE solo una proposta che l'altra parte ha registrato tramite PROPOSTA.",
        "You may only REFUSE a proposal the other party has logged via PROPOSAL.": "Puoi RIFIUTARE solo una proposta che l'altra parte ha registrato tramite PROPOSTA.",
        "Total effort of any PROPOSAL or AGREE set must be ≤ LIMIT.": "Lo sforzo totale di qualsiasi set PROPOSTA o ACCORDO deve essere ≤ LIMITE.",
        "Do NOT reveal your hidden importance scores.": "NON rivelare i tuoi punteggi di importanza nascosti.",
        "A tag in a structured format must be followed by colon and whitespace. The argument must be a python set containing 0 or more strings.":
            "Un tag in un formato strutturato deve essere seguito da due punti e uno spazio. L'argomento deve essere un set python contenente 0 o più stringhe.",
        "So, it must be of the form TAG: {...}": "Quindi, deve essere della forma TAG: {...}",
        "Strictly follow the interaction protocol and DO NOT write anything beyond the given structure.": "Segui rigorosamente il protocollo di interazione e NON scrivere nulla al di fuori della struttura data.",
        "The game ends when one side gives an AGREE to a PROPOSAL made by the other player.": "Il gioco termina quando una parte dà un ACCORDO a una PROPOSTA fatta dall'altro giocatore.",
        "The content in your response which can be handed to the other player has to be non-empty.": "Il contenuto della tua risposta che può essere consegnato all'altro giocatore non deve essere vuoto.",
        "Only proposals which have been logged via the PROPOSAL format structure and which haven't been refused via REFUSE are active.":
            "Sono attive solo le proposte registrate tramite la struttura del formato PROPOSTA e che non sono state rifiutate tramite RIFIUTO.",
        "You must include the ARGUMENT format at least once somewhere in all of your messages.": "Devi includere il formato ARGOMENTO almeno una volta da qualche parte in tutti i tuoi messaggi.",
        "You must include the STRATEGIC REASONING format only once at the very beginning of every one of your messages and not more often. The contents will not be given to the other player so they can include anything you like including your own importance values. Here you should reason step by step to come up with you next move.":
            "Devi includere il formato RAGIONAMENTO STRATEGICO solo una volta all'inizio di ogni tuo messaggio e non più spesso. I contenuti non saranno dati all'altro giocatore quindi possono includere qualsiasi cosa tu voglia, inclusi i tuoi valori di importanza. Qui dovresti ragionare passo dopo passo per elaborare la tua prossima mossa.",
        "You will now receive the first message of the other player.": "Ora riceverai il primo messaggio dell'altro giocatore.",

        # Tags with colons (for prompts and mock responses)
        "STRATEGIC REASONING:": "RAGIONAMENTO STRATEGICO:",
        "ARGUMENT:": "ARGOMENTO:",
        "PROPOSAL:": "PROPOSTA:",
        "AGREE:": "ACCORDO:",
        "REFUSE:": "RIFIUTO:",

        # Standalone Tags (for regex patterns and general text)
        "STRATEGIC REASONING": "RAGIONAMENTO STRATEGICO",
        "ARGUMENT": "ARGOMENTO",
        "PROPOSAL": "PROPOSTA",
        "AGREE": "ACCORDO",
        "REFUSE": "RIFIUTO",

        # General Keywords
        "LIMIT": "LIMITE",
        "Goal:": "Obiettivo:",
        "Rules:": "Regole:",
        "Interaction Protocol:": "Protocollo di Interazione:",
        "Item effort": "Sforzo oggetto",
        "Item importance values": "Valori di importanza dell'oggetto"


    }

    def replace_text(text, trans_dict):
        """Helper function to replace all occurrences in a string."""
        for eng, ita in trans_dict.items():
            text = text.replace(eng, ita)
        return text

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{input_filepath}' is not a valid JSON file.")
        return

    # List of keys whose string values should be translated.
    keys_to_translate_content = [
        "refuse_error_prompt", "agreement_ambiguous_prompt", "agreement_non_active_prompt",
        "proposal_error_prompt", "invalid_python_set_error", "no_sr_prompt", "no_argument_prompt",
        "untagged_sequence_prompt", "only_sr_prompt", "player1_initial_prompt",
        "player2_initial_prompt", "mock_response_p1", "mock_response_p2"
    ]

    # Map of regex tag keys to their English keywords
    regex_tags_map = {
        "agree_tag": "AGREE",
        "refusal_tag": "REFUSE",
        "proposal_tag": "PROPOSAL",
        "argument_tag": "ARGUMENT",
        "strategic_reasoning_tag": "STRATEGIC REASONING"
    }

    # Iterate through the JSON structure and translate the content.
    for experiment in data.get("experiments", []):
        for instance in experiment.get("game_instances", []):
            # 1. Translate descriptive content
            for key in keys_to_translate_content:
                if key in instance and isinstance(instance[key], str):
                    instance[key] = replace_text(instance[key], translations)

            # 2. Translate regex patterns
            for key, keyword in regex_tags_map.items():
                if key in instance and isinstance(instance[key], str):
                    translated_keyword = translations.get(keyword, keyword)
                    instance[key] = instance[key].replace(keyword, translated_keyword)

    # Write the translated data to the output file.
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Full translation complete. The translated file has been saved as '{output_filepath}'")


# --- To run the script ---
# 1. Save the code above as a Python file (e.g., `translate_script.py`).
# 2. Make sure the `instances_it.json` file is in the same directory.
# 3. Run the script from your terminal: `python translate_script.py`
if __name__ == '__main__':
    translate_json_content_fully(input_filepath='clembench/air_balloon_survival/in/instances_en.json', output_filepath='clembench/air_balloon_survival/in/instances_it.json')