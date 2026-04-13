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
            "Sklasyfikuj intencję użytkownika do jednej z kategorii: "
            "[GREETING, SITUATION, QUESTION, SMALL_TALK, SEARCH_TOPIC].\n"
            "Użyj SEARCH_TOPIC TYLKO wtedy, gdy użytkownik wprost prosi o tekst na konkretny temat "
            "(np. 'Daj jakiś wers o policji', 'Zarzuć tekstem o miłości').\n"
            "Zwróć TYLKO słowo klucz.\n"
            f"User: {user_input}"
        )
        response = ollama.generate(model=MODEL_MAIN, prompt=prompt, options={'temperature': 0.0})
        intent = response['response'].strip().upper()

        print(intent)

        return intent if intent in ["GREETING", "SITUATION", "QUESTION", "SEARCH_TOPIC"] else "SITUATION"

    def extract_search_tag(self, user_input):
        prompt = (
            "Użytkownik szuka cytatu na konkretny temat. Wyciągnij GŁÓWNY temat jako JEDNO słowo "
            "(w mianowniku, np. 'miłość', 'policja', 'pieniądze').\n"
            f"Tekst: {user_input}"
        )
        response = ollama.generate(model=MODEL_MAIN, prompt=prompt, options={'temperature': 0.0})
        return response['response'].strip().lower()

    def generate_hyde_answer(self, user_input, intention, tag):
        if intention == "SEARCH_TOPIC":
            return ""

        style_guide = {
            "GREETING": "Odpowiedz jak ziomek na osiedlu, przywitaj się, zapytaj co u niego.",
            "SITUATION": f"Odpowiedz jak starszy brat, daj radę albo skomentuj to w ulicznym stylu. Użyj słowa: {tag}",
            "QUESTION": f"Odpowiedz konkretnie, ale używając slangowych metafor. Użyj słowa: {tag}",
            "SMALL_TALK": "Odpowiedź jakimś luźnym tekstem nawiązujacym do otrzymanego tekstu."
        }

        prompt = (
            "Jesteś polskim raperem.\n"
            f"Napisz krótką, 2-wersową odpowiedź rymowaną. {style_guide.get(intention)}\n"
            "Nie dodawaj żadnych swoich wstawek, masz wypluć sam rym.\n"
            f"User powiedział: {user_input}"
        )

        res = ollama.generate(
            model=MODEL_HELPER,
            prompt=prompt,
            options={'temperature': 0.8}
        )
        return res['response'].strip()

    def get_context(self, user_input, hyde_res, intention, tag):
        res_hyde, res_orig = None, None


        if intention == "SEARCH_TOPIC":
            try:
                res_orig = self.collection.query(
                    query_texts=[f"query: {tag}"],
                    n_results=15,
                    where={"tags": {"$contains": tag}}
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

        if not candidates: return "Brak wyników."

        query_for_reranker = hyde_res if hyde_res else user_input
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