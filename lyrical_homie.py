import re

import ollama
from pathlib import Path
import chromadb
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('sdadas/polish-reranker-roberta-v3')
MODEL_MAIN = 'qwen2.5:7b-instruct'
MODEL_HELPER = 'SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M'


class LyricalEngine:
    def __init__(self, db_path):
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_collection(name="polish_rap_lyrics")

    def classify_intent(self, user_input):
        prompt = (
            "Postaraj się sklasyfikować tekst użytkownika do jednej z poniższych kategorii: "
            "[GREETING, SITUATION, QUESTION, BRAGGA, SEARCH_TOPIC].\n"
            "GREETING - powitanie, pozdrowienie lub życzenia.\n"
            "SITUATION - problem użytkownika, sprawa wymagjąca porady.\n"
            "QUESTION - pytanie, wątpliwość (wymagany znak zapytania)\n"
            "BRAGGA - luźny tekst, nie do końca poważny, nie pasujący do żadnej z wyżej wymienionych kategorii.\n"
            "SEARCH_TOPIC - użyj TYLKO wtedy, gdy użytkownik wprost prosi o tekst na konkretny temat "
            "(np. 'Daj jakiś wers o policji', 'Zarzuć tekstem o miłości').\n"
            "Zwróć TYLKO słowo klucz.\n"
            f"Użytkownik napisał: {user_input}"
        )
        response = ollama.generate(
            model=MODEL_MAIN,
            prompt=prompt,
            options={'temperature': 0.0}
        )
        intent = response['response'].strip().upper()

        print(intent)

        return intent if intent in ["GREETING", "SITUATION", "QUESTION", "BRAGGA", "SEARCH_TOPIC"] else "SITUATION"

    def extract_search_tag(self, user_input):
        prompt = (
            "Użytkownik opisuje sytuację. Wyciągnij od 1 do maksymalnie 3 najważniejszych słów kluczowych, "
            "które idealnie opisują ten problem (np. 'dziewczyna', 'rozstanie', 'zdrada').\n"
            "ZASADY:\n"
            "1. Używaj TYLKO poprawnych polskich słów (najlepiej rzeczowników w mianowniku).\n"
            "2. Nie wymyślaj własnych słów, używaj form słownikowych.\n"
            "3. Zwróć same słowa oddzielone przecinkami.\n"
            f"Tekst: {user_input}"
        )
        response = ollama.generate(model=MODEL_MAIN, prompt=prompt, options={'temperature': 0.1})
        return response['response'].strip().lower()


    def generate_hyde_answer(self, user_input, intention, tag):

        if intention == "SEARCH_TOPIC":
            return ""

        style_guide = {
            "GREETING": (
                "STYL: Luźna gadka, osiedlowa duma. "
                "PRZYKŁAD: 'Siema mordo, dobrze cię widzieć na rewirze. Co tam u ciebie słychać, jak życie płynie?'"
            ),
            "SITUATION": (
                "STYL: Surowy, życiowy autentyzm, ból i duma. Zero litości, sama prawda. "
                f"TEMAT: {tag}. PRZYKŁAD: 'Życie to nie bajka, znowu dostajesz po plecach. Ale stój kurwa twardo, ziomek, bo szacunek to forteca.'"
            ),
            "QUESTION": (
                "STYL: Konkret, rapowa metafora, bez owijania w bawełnę. "
                f"TEMAT: {tag}. PRZYKŁAD: 'Pytasz o drogę? Tu mapą jest serce i lojalność. Reszta to tylko tło, tania teatralność.'"
            ),
            "BRAGGA": (
                "STYL: Pewność siebie, luksus, agresywny sukces. "
                "PRZYKŁAD: 'Wjeżdżam w to miasto, złoto na szyi się świeci. Mam drogie buty, czas na Rolex szybko leci.'"
            )
        }

        chosen_style = style_guide.get(intention, "Napisz luźny, rapowy komentarz do tej sytuacji.")

        prompt = (
            "Jesteś polskim raperem. Twoim zadaniem jest napisanie dwóch  zdań, które posłużą jako odpowiedż na tekst.\n\n"
            f"INPUT UŻYTKOWNIKA na który odpowiadasz: {user_input}\n"
            f"{chosen_style} \n"
            "ZASADY:\n"
            "1. Napisz DOKŁADNIE 2 (DWA) mocne, KRÓTKIE i dosadne zdania.\n"
            "2. Używaj ulicznego slangu, polskiego rapowego słownictwa.\n"
            "3. Nie bój się być wulgarny, jeśli sprawa tego wymaga.\n"
            "RAP:"
        )

        try:
            res = ollama.generate(
                model=MODEL_HELPER,
                prompt=prompt,
                options={'temperature': 0.6}
            )
            return res['response'].strip()
        except Exception as e:
            print(f"   [Błąd HyDE: {e}]")
            return ""

    def get_context(self, user_input, hyde_res, intention, tag):
        res_hyde, res_orig = None, None


        if intention == "SEARCH_TOPIC":
            try:
                res_orig = self.collection.query(
                    query_texts=[f"query: {tag}"],
                    n_results=15,
                )
            except Exception:
                res_orig = self.collection.query(query_texts=[f"query: {user_input}"], n_results=15)
        else:
            res_hyde = self.collection.query(query_texts=[f"query: {hyde_res}"], n_results=15)
            res_orig = self.collection.query(query_texts=[f"query: {user_input}"], n_results=10)

        seen_parents = set()
        candidates = []

        results_to_process = [r for r in [res_hyde, res_orig] if r is not None]

        for res in results_to_process:
            if not res['documents'] or not res['documents'][0]:
                continue

            for i in range(len(res['documents'][0])):
                doc = res['documents'][0][i]
                meta = res['metadatas'][0][i]
                parent = meta['parent_text']

                if parent not in seen_parents:
                    seen_parents.add(parent)
                    tags = meta.get('tags', '')
                    if isinstance(tags, list):
                        tags_display = ", ".join(tags)
                    else:
                        tags_display = str(tags)

                    tag_prefix = f"[Tagi: {tags_display}] " if tags_display else ""
                    tagged_parent = f"{tag_prefix}{parent}"

                    candidates.append({
                        "child": doc,
                        "parent": parent,
                        "tagged_parent": tagged_parent,
                        "artist": meta['artist'],
                        "title": meta['title'],
                        "tags": tags_display
                    })
        if not candidates:
            return "Brak wyników."

        if hyde_res:
            query_for_reranker = f"Użytkownik pisze: '{user_input}'. Oczekiwana odpowiedź w stylu: '{hyde_res}'"
        else:
            query_for_reranker = user_input

        pairs = [[query_for_reranker, c['tagged_parent']] for c in candidates]


        scores = reranker.predict(pairs)

        for idx, score in enumerate(scores):
            candidates[idx]['score'] = score

        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        return sorted_candidates[:3]

    def final_response(self, user_input, candidates):

        context_str = "\n".join([
            f"ID: {i} | [{c['artist']} - {c['title']}] [Temat/Tagi: {c['tags']}]\nLiryka:\n{c['parent']}\n---"
            for i, c in enumerate(candidates)
        ])

        system_prompt = (
            "Jesteś 'Lyrical Homie'. Rozmawiasz z ziomkiem używając wyłącznie cytatów z polskiego rapu.\n"
            "ZASADY:\n"
            "1. Otrzymujesz cytaty wzbogacone o [Temat/Tagi]. Użyj tych tagów, żeby zrozumieć vibe utworu.\n"
            "2. Wybierz jeden, najbardziej pasujący cytat do sytuacji lub pytania.\n"
            "3. WYTNIJ z niego TYLKO 1 lub maksymalnie 2 najlepsze, najbardziej trafne wersy (punchline), które idealnie odpowiadają na słowa ziomka.\n"
            "4. Zwróć TYLKO wybrany cytat (same wersy).\n"
            "5. Nie podawaj autora, tytułu, ani tagów w odpowiedzi."
        )

        user_prompt = (
            f"Ziomek mówi: {user_input}\n\n"
            f"DOSTĘPNE CYTATY:\n{context_str}"
        )

        res = ollama.chat(
            model=MODEL_MAIN,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={'temperature': 0.2}
        )
        return res['message']['content']

    def final_response_parent_retrieval(self, user_input, candidates):
        all_pairs = []

        for c in candidates:
            lines = [line.strip() for line in c['parent'].split('\n') if line.strip()]

            if len(lines) <= 4:
                all_pairs.append({
                    "text": "\n".join(lines),
                    "artist": c['artist'],
                    "title": c['title']
                })
            else:
                # Wystarczy odjąć 3, żeby pętla wiedziała gdzie się zatrzymać
                for i in range(len(lines) - 3):
                    # Slicing bierze 4 linijki naraz (od i do i+4) i od razu skleja enterami!
                    verses = "\n".join(lines[i:i + 4])
                    all_pairs.append({
                        "text": verses,
                        "artist": c['artist'],
                        "title": c['title']
                    })

        unique_pairs_dict = {pair['text']: pair for pair in all_pairs}
        unique_pairs = list(unique_pairs_dict.values())

        context_str = "\n\n".join([
            f"OPCJA {i + 1}:\n{p['text']}"
            for i, p in enumerate(unique_pairs)
        ])

        system_prompt = (
            "Jesteś 'Lyrical Homie'. Udzielasz rapowych ripost.\n"
            "ZASADY:\n"
            "1. Otrzymujesz listę kilkunastu 4-wersowych opcji.\n"
            "2. WYBIERZ TYLKO JEDNĄ najlepszą opcję, która idealnie pasuje do słów użytkownika.\n"
            "3. Wybieraj opcje, które najbardziej merytorycznie odpowiadają na zadany temat.\n"
            "4. Zwróć WYŁĄCZNIE DOKŁADNY TEKST wybranej opcji. Żadnych wstępów, numerów opcji, ani komentarzy."
        )

        user_prompt = (
            f"Użytkownik mówi: {user_input}\n\n"
            f"DOSTĘPNE OPCJE DO WYBORU:\n{context_str}"
        )

        res = ollama.chat(
            model=MODEL_MAIN,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={'temperature': 0.1}
        )

        return res['message']['content'].strip()

    def final_response_child_retrieval(self, user_input, candidates):
        clean_options = []

        for c in candidates:
            raw_child = c['child']

            raw_child = raw_child.replace("passage:", "")
            clean_text = re.sub(r'\[Tagi:.*?\]\s*', '', raw_child).strip()

            clean_options.append(clean_text)

        context_str = "\n\n".join([f"OPCJA {i + 1}:\n{text}" for i, text in enumerate(clean_options)])

        system_prompt = (
            "Jesteś 'Lyrical Homie'. Udzielasz rapowych ripost.\n"
            "ZASADY:\n"
            "1. Otrzymujesz listę TRZECH opcji.\n"
            "2. WYBIERZ TYLKO JEDNĄ najlepszą opcję, która idealnie pasuje do słów użytkownika.\n"
            "3. Zwróć WYŁĄCZNIE DOKŁADNY TEKST wybranej opcji. Żadnych wstępów, numerów opcji, ani komentarzy."
        )

        user_prompt = (
            f"Użytkownik mówi: {user_input}\n\n"
            f"DOSTĘPNE OPCJE DO WYBORU:\n{context_str}"
        )

        res = ollama.chat(
            model=MODEL_MAIN,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={'temperature': 0.1}
        )
        x = res['message']['content'].strip()
        print(x)
        return x
if __name__ == "__main__":
    engine = LyricalEngine(Path(__file__).parent / "vector_db")

    print("Elo! Co tam u Ciebie mordeczko?")
    while True:
        u_msg = input("Ty: ")
        if u_msg.lower() in ['exit', 'pa', 'nara', 'quit']: break

        intent = engine.classify_intent(u_msg)
        target_tag = engine.extract_search_tag(u_msg)
        hyde_text = engine.generate_hyde_answer(u_msg, intent, target_tag)
        top_candidates = engine.get_context(u_msg, hyde_text, intent, target_tag)


        if top_candidates == "Brak wyników.":
            print("\nLyrical Homie: Sory ziomek, pusta głowa, nie mam do tego rymu.\n")
            continue

        reply = engine.final_response(u_msg, top_candidates)
        print(f"\nLyrical Homie: {reply}\n")