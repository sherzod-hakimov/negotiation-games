import json
import re


def translate_json_to_italian(input_file, output_file):
    """
    Translates English content in JSON file to Italian while preserving structure and formatting.
    """

    # Translation dictionary - English to Italian
    translations = {
        # Error messages and penalties
        "Penalty: $reason": "Penalità: $reason",
        "Make sure that your response only contains either `SAY: <MESSAGE>` or `MOVE: <OBJECT>, (<X>, <Y>)`, and nothing else!": "Assicurati che la tua risposta contenga solo `DICI: <MESSAGE>` o `SPOSTA: <OBJECT>, (<X>, <Y>)`, e nient'altro!",
        "Your message has been relayed to the other player.": "Il tuo messaggio è stato inoltrato all'altro giocatore.",
        "What is your next command?": "Qual è il tuo prossimo comando?",
        "The other player moved an object on their grid.": "L'altro giocatore ha spostato un oggetto sulla sua griglia.",
        "The other player sent this message:": "L'altro giocatore ha inviato questo messaggio:",
        "You have collectively accumulated $penalty of 8 penalties.": "Avete accumulato collettivamente $penalty di 8 penalità.",
        "Please try again!": "Riprova!",
        "You are currently playing round $round of maximum 12.": "Stai attualmente giocando il round $round di massimo 12.",

        # Parse errors
        "Your message must not contain anything before the command!": "Il tuo messaggio non deve contenere nulla prima del comando!",
        "Your message must not contain anything after the command!": "Il tuo messaggio non deve contenere nulla dopo il comando!",
        "Your message must not contain anything before or after the command!": "Il tuo messaggio non deve contenere nulla prima o dopo il comando!",
        "Your message contains more than one command!": "Il tuo messaggio contiene più di un comando!",
        "Your message contains restricted content!": "Il tuo messaggio contiene contenuti riservati!",
        "Your message is not in the expected format!": "Il tuo messaggio non è nel formato previsto!",
        "You must begin the game by sending a message to the other player!": "Devi iniziare il gioco inviando un messaggio all'altro giocatore!",

        # Move messages
        "Moved '$object' to ($x,$y) successfully. Your updated grid looks like this:": "Spostato '$object' in ($x,$y) con successo. La tua griglia aggiornata appare così:",
        "Invalid move: ($x,$y) is out of bounds. Please try again!": "Mossa non valida: ($x,$y) è fuori dai limiti. Riprova!",
        "Penalty: ($x,$y) is not empty, but contains '$object'.": "Penalità: ($x,$y) non è vuoto, ma contiene '$object'.",
        "Invalid move: Your image has no object with ID '$object'. Please try again!'. Please try again!": "Mossa non valida: La tua immagine non ha nessun oggetto con ID '$object'. Riprova!",



        # Game prompts - Player 1
        "I am your game master, and you are playing a collaborative game with the following grid as game board:": "Sono il tuo game master, e stai giocando un gioco collaborativo con la seguente griglia come tavolo da gioco:",
        "The upper edge displays x-coordinates increasing to the right, and the right edge y-coordinates increasing downward.": "Il bordo superiore mostra le coordinate x che aumentano verso destra, e il bordo destro le coordinate y che aumentano verso il basso.",
        "The following objects are randomly placed on your grid: 'C', 'L', 'P'.": "I seguenti oggetti sono posizionati casualmente sulla tua griglia: 'C', 'L', 'P'.",
        "The other player sees a variation of the game board, where the objects are placed at different random locations. You cannot see the other player's board, and they cannot see yours.": "L'altro giocatore vede una variante del tavolo da gioco, dove gli oggetti sono posizionati in diverse posizioni casuali. Non puoi vedere il tavolo dell'altro giocatore, e loro non possono vedere il tuo.",
        "Both players need to move the objects on their respective background so that identical objects end up at the same coordinates. You have to communicate with the other player to agree upon a common goal configuration.": "Entrambi i giocatori devono spostare gli oggetti sui rispettivi sfondi in modo che oggetti identici finiscano alle stesse coordinate. Devi comunicare con l'altro giocatore per concordare una configurazione obiettivo comune.",
        "In each turn, you can send exactly one of the following two commands:": "In ogni turno, puoi inviare esattamente uno dei seguenti due comandi:",
        "`SAY: <MESSAGE>`: to send a message (everything up until the next line break) to the other player. I will forward it to your partner.\n2. `MOVE: <OBJECT>, (<X>, <Y>)`: to move an object to a new position": "`DICI: <MESSAGE>`: per inviare un messaggio (tutto fino alla prossima interruzione di riga) all'altro giocatore. Lo inoltrerò al tuo partner.\n2. `SPOSTA: <OBJECT>, (<X>, <Y>)`: per spostare un oggetto in una nuova posizione",
        "to move an object to a new position, where `<X>` is the column and `<Y>` is the row. I will inform you if your move was successful or not.": "per spostare un oggetto in una nuova posizione, dove `<X>` è la colonna e `<Y>` è la riga. Ti informerò se la tua mossa è stata riuscita o meno.",
        "If you don't stick to the format, or send several commands at once, I have to penalize you.": "Se non rispetti il formato, o invii diversi comandi contemporaneamente, devo penalizzarti.",
        "If both players accumulate more than 8 penalties, you both lose the game.": "Se entrambi i giocatori accumulano più di 8 penalità, perdete entrambi il gioco.",
        "It is vital that you communicate with the other player regarding your goal state! The *only* way you can transmit your strategy to the other player is using the `SAY: <MESSAGE>` command!": "È vitale che comunichi con l'altro giocatore riguardo al tuo stato obiettivo! L'*unico* modo in cui puoi trasmettere la tua strategia all'altro giocatore è usando il comando `DICI: <MESSAGE>`!",
        "You can only move objects to cells within the bounds of the grid. The target cell must be empty, i.e., it must only contain the symbol '◌'.": "Puoi spostare oggetti solo in celle entro i limiti della griglia. La cella di destinazione deve essere vuota, cioè deve contenere solo il simbolo '◌'.",
        "If you try to move an object to a spot that is not empty, or try to move it outside of the grid, I have to penalize you. You get another try.": "Se provi a spostare un oggetto in un posto che non è vuoto, o provi a spostarlo fuori dalla griglia, devo penalizzarti. Hai un altro tentativo.",
        "Before making a move, double check that the target spot is empty, and does not hold any letter, frame, or line!": "Prima di fare una mossa, controlla due volte che il posto di destinazione sia vuoto, e non contenga alcuna lettera, cornice, o linea!",
        "If you think you reached the goal of aligning all objects, you can ask the other player to finish the game by sending `SAY: finished?`. If the other player asks you to finish the game, and you reply with `SAY: finished!`": "Se pensi di aver raggiunto l'obiettivo di allineare tutti gli oggetti, puoi chiedere all'altro giocatore di finire il gioco inviando DICI: finito?. Se l'altro giocatore ti chiede di finire il gioco, e rispondi con DICI: finito!",
        "Both players win if the game ends within 12 rounds, where one round is defined as two players each sending a valid command.": "Entrambi i giocatori vincono se il gioco finisce entro 12 round, dove un round è definito come due giocatori che inviano ciascuno un comando valido.",
        "The closer the identical objects are in both game boards, the more points you get. Penalties reduce your points. Can you beat the record?": "Più vicini sono gli oggetti identici in entrambi i tavoli da gioco, più punti ottieni. Le penalità riducono i tuoi punti. Riesci a battere il record?",
        "Please send a message to the other player to start the game!": "Per favore invia un messaggio all'altro giocatore per iniziare il gioco!",
        "where `<X>` is the column and `<Y>` is the row. I will inform you if your move was successful or not.": "dove <X> è la colonna e <Y> è la riga. Ti farò sapere se la tua mossa è andata a buon fine o no.",

        # Termination
        "finished?": "finito?",
        "finished!": "finito!",

        # Player 2 specific
        "The other player started the game by sending this message:": "L'altro giocatore ha iniziato il gioco inviando questo messaggio:",
        "What is your first command?": "Qual è il tuo primo comando?",

        # Section headers
        "**Goal:**": "**Obiettivo:**",
        "**Rules:**": "**Regole:**",
        "**Moving Objects**": "**Spostamento Oggetti**",
        "**End of Game**": "**Fine del Gioco**",
        "**Scoring:**": "**Punteggio:**"
    }

    def translate_text(text):
        """Translate a text string using the translations dictionary."""
        if not isinstance(text, str):
            return text

        translated_text = text
        for english, italian in translations.items():
            translated_text = translated_text.replace(english, italian)

        return translated_text

    def translate_dict(obj):
        """Recursively translate dictionary values."""
        if isinstance(obj, dict):
            translated = {}
            for key, value in obj.items():
                # Special handling for regex pattern keys
                if key in ["say_pattern", "move_pattern"]:
                    if key == "say_pattern":
                        translated[key] = "(?P<head>.*)DICI: (?P<message>[^\\n]+)(?P<tail>.*)"
                    elif key == "move_pattern":
                        translated[
                            key] = "(?P<head>.*)SPOSTA: (?P<obj>[^,]+), \\((?P<x>\\d+), ?(?P<y>\\d+)\\)(?P<tail>.*)"
                else:
                    translated[key] = translate_dict(value)
            return translated
        elif isinstance(obj, list):
            return [translate_dict(item) for item in obj]
        elif isinstance(obj, str):
            return translate_text(obj)
        else:
            return obj

    # Read the input JSON file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Translate the content
        translated_data = translate_dict(data)

        # Write the translated JSON to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

        print(f"Translation completed successfully!")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file '{input_file}'.")
    except Exception as e:
        print(f"Error: {str(e)}")


# Example usage
if __name__ == "__main__":
    input_filename = "clembench/clean_up/in/instances.json"  # Change this to your input file
    output_filename = "clembench/clean_up/in/instances_it.json"  # Change this to your desired output file

    translate_json_to_italian(input_filename, output_filename)