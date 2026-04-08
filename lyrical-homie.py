import json
import re
from pathlib import Path
import chromadb
import numpy as np
import ollama
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('sdadas/polish-reranker-roberta-v3')

BASE_DIR = Path(__file__).parent


CHROMA_PATH = BASE_DIR / "vector_db"
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name="polish_rap_lyrics")

def generate_tag(text):
    prompt = (
        "Jesteś analitykiem. Twoim jedynym zadaniem jest wyciągnięcie GŁÓWNEJ emocji"
        "poniższego tekstu. Używaj oficjalnych, słownikowych pojęć. Zwróć TYLKO jedno słowo."
        "Zwróć plik JSON z jednym kluczem 'tag'.\n\n"
        f"Tekst:\n{text}"
    )

    try:
        response = ollama.generate(
            model='SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M',
            prompt=prompt,
            format='json',
            options={'temperature': 0.0}
        )

        data = json.loads(response['response'])
        tag = data.get('tag', [])
        return tag

    except Exception as e:
        print(f"   [Błąd LLM: {e}]")
        return ""

def get_lyrics_context(user_input, hypothetical_rap, target_tag=None):
    q_hyde_prefix = f"query: {hypothetical_rap}"
    q_user_prefix = f"query: {user_input}"

    results_hyde = collection.query(
        query_texts=[q_hyde_prefix],
        n_results=20
    )
    results_user = collection.query(
        query_texts=[q_user_prefix],
        n_results=20
    )

    all_query_results = [results_hyde, results_user]

    if target_tag:
        try:
            results_hyde_tagged = collection.query(
                query_texts=[q_hyde_prefix],
                n_results=10,
                where={"tags": {"$contains": target_tag.lower()}}
            )
            results_user_tagged = collection.query(
                query_texts=[q_user_prefix],
                n_results=10,
                where={"tags": {"$contains": target_tag.lower()}}
            )
            all_query_results.extend([results_hyde_tagged, results_user_tagged])
        except Exception as e:
            pass  # Zabezpieczenie, gdyby baza wektorowa nie znalazła żadnego tagu

    # 3. Zbieranie i deduplikacja (żeby Cross-Encoder nie liczył dwa razy tego samego)
    all_docs = []
    all_children = []
    all_metas = []
    seen = set()

    for res in all_query_results:
        # ChromaDB zwraca listy list, więc upewniamy się, że są jakiekolwiek dokumenty
        if not res['documents'] or not res['documents'][0]:
            continue

        for i, doc in enumerate(res['documents'][0]):
            parent_doc = res['metadatas'][0][i]['parent_text']
            child_doc = doc.replace("passage: ", "").strip()

            if parent_doc not in seen:
                seen.add(parent_doc)
                all_docs.append(parent_doc)
                all_children.append(child_doc)
                all_metas.append(res['metadatas'][0][i])

    # 4. Cross-Encoder (Reranking) - On podejmuje ostateczną decyzję!
    combined_query = f"SYTUACJA: {user_input}. OCZEKIWANY STYL: {hypothetical_rap}"
    cross_inp = [[combined_query, parent] for parent in all_docs]

    # Jeśli nie ma nic do rerankowania, zwracamy pusty string
    if not cross_inp:
        return "Brak wyników."

    scores = reranker.predict(cross_inp)

    sorted_indices = np.argsort(scores)[::-1]
    top_k = min(5, len(sorted_indices))  # Zabezpieczenie, gdyby znaleziono mniej niż 5
    best_indices = sorted_indices[:top_k]

    # 5. Formatowanie wyników
    context = ""
    for idx in best_indices:
        best_2_lines = all_children[idx]
        meta = all_metas[idx]
        tags_display = meta.get('tags', '')

        # Wyświetlamy tagi tylko jeśli istnieją, żeby ładnie wyglądało w logach/odpowiedzi
        tag_str = f" [Tagi: {tags_display}]" if tags_display else ""

        context += f"[{meta['artist']} - {meta['title']}]{tag_str}\n{best_2_lines}\n---\n"
        # context += f"Wynik: {scores[idx]:.4f} \n\n" # Zakomentowałem wynik dla czystości promptu końcowego

    return context




def lyrical_chat(user_input):
    hyde_system_instructions = (
        "Jesteś systemem, który odpowiada na zdanie użytkownika składając dwa uliczne wersy polskiego rapu.\n"
        "ZASADA 1: Napisz DOKŁADNIE DWA wersy.\n"
        "ZASADA 2: Nie dodawaj absolutnie żadnych wstępów.\n"
        "ZASADA 3: Nie numeruj wersów.\n"
        "ZASADA 4: Od razu zacznij pisać rymy.\n"
        "ZASADA 5: Możesz być wulgarny i chamski.\n"
        "PRZYKŁAD: \n"
        "Zdanie użytkownika: Kumpel oszukał mnie na hajs.\n"
        "Twoja odpowiedź:\n"
        "Fałszywy ziomek, w oczach tylko plik,\n"
        "Zabrał flotę, teraz dla mnie to jest nikt."
    )

    hyde_response = ollama.chat(
        model='SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M',
        messages=[
            {'role': 'system', 'content': hyde_system_instructions},
            {'role': 'user', 'content': user_input},
        ],
        options={
            'temperature': 0.6,
            'stop': ['<|im_id|>', '<|im_end|>', '<|im_sep|>']
        }
    )

    tag = generate_tag(user_input)
    hypothetical_rap = hyde_response['message']['content'].strip()
    context = get_lyrics_context(user_input, hypothetical_rap, tag)



    system_instruction = (
        "Jesteś 'Lyrical Homie'. Odpowiadasz WYŁĄCZNIE cytatami z polskiego rapu.\n\n"
        "INSTRUKCJA:\n"
        "1. Otrzymasz poniżej kilka cytatów z polskiego rapu i 1 opis sytuacji ziomka.\n"
        "2. Wybierz TYLKO JEDEN cytat, który najlepiej odnosi się do opisanej sytuacji.\n"
        "3. Upewnij się, że wybrany cytat stanowi logiczną ODPOWIEDŹ na problem ziomka.\n"
        "4. NIE PISZ absolutnie nic więcej. Żadnych wstępów, pozdrowień ani własnych komentarzy."
    )

    prompt = (
        f"OPIS SYTUACJI ZIOMKA: {user_input}\n\n" 
        f"DOSTĘPNE CYTATY Z BAZY:\n{context}\n\n"
        "Sformułuj swoją odpowiedź wybierając odpowiedni fragment z bazy, dokładnie według wytycznych z INSTRUKCJI."
    )

    response = ollama.chat(
        model='qwen2.5:7b-instruct',
        messages=[
            {'role': 'system', 'content': system_instruction},
            {'role': 'user', 'content': prompt},
        ],
        options={
            'temperature': 0.0,
            'stop': ['<|im_id|>', '<|im_end|>', '<|im_sep|>']
        }
    )

    return response['message']['content']

if __name__ == "__main__":
    print("Elo! Jestem Twoim muzycznym wsparciem. Co u Ciebie?")
    while True:
        user_msg = input("Ty: ")
        if user_msg.lower() in ['exit', 'quit', 'pa']: break

        reply = lyrical_chat(user_msg)
        clean_reply = result = re.sub(r"\[.*?\]", "", reply)
        print(f"\nLyrical Homie:\n{clean_reply}\n")