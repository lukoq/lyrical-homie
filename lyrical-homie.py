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


def get_lyrics_context(user_input, hypothetical_rap):
    results_hyde = collection.query(
        query_texts=[hypothetical_rap],
        n_results=10
    )
    results_user = collection.query(
        query_texts=[user_input],
        n_results=10
    )

    all_docs = []
    all_metas = []
    seen = set()

    for res in [results_hyde, results_user]:
        for i, doc in enumerate(res['documents'][0]):
            if doc not in seen:
                seen.add(doc)
                all_docs.append(doc)
                all_metas.append(res['metadatas'][0][i])

    cross_inp = [[user_input, doc] for doc in all_docs]
    scores = reranker.predict(cross_inp)

    sorted_indices = np.argsort(scores)[::-1]
    top_k = 5
    best_indices = sorted_indices[:top_k]

    context = ""
    for idx in best_indices:
        doc = all_docs[idx]
        meta = all_metas[idx]
        context += f"[{meta['artist']} - {meta['title']}]: {doc}\n---\n"

    return context




def lyrical_chat(user_input):
    hyde_prompt = (
        f"Napisz krótkie, 2-wersowe nawinięcie w stylu polskiego rapu, "
        f"które pasuje do sytuacji: '{user_input}'. "
        f"Ma to być w stylu chwytliwego tekstu. Możesz użyć wulgaryzmów."
        f"Napisz SAME rymy, bez żadnych wstępów czy komentarzy."
    )

    hyde_response = ollama.chat(
        model='SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M',
        messages=[{'role': 'user', 'content': hyde_prompt}],
        options={'temperature': 0.7}
    )

    hypothetical_rap = hyde_response['message']['content'].strip()
    context = get_lyrics_context(user_input, hypothetical_rap)

    system_instruction = (
        "Jesteś 'Lyrical Homie'. Odpowiadasz WYŁĄCZNIE cytatami z polskiego rapu.\n\n"
        "ZASADY:\n"
        "1. WYBIERZ DWA WERSY (linijki) z jednego, najbardziej trafnego utworu.\n"
        "2. MUSISZ ODPOWIADAĆ TYLKO W JĘZYKU POLSKIM. Ignoruj teksty słowackie, czeskie lub angielskie, nawet jeśli są w bazie.\n"
        "3. WYPLUWAJ TYLKO CZYSTY TEKST CYTATU. Zero komentarza, zero tytułu, zero wykonawcy.\n"
        "4. FORMAT: Linijka 1\nLinijka 2"
    )

    prompt = (
        f"SYTUACJA ZIOMKA: {user_input}\n\n"
        f"DOSTĘPNE WERSY Z BAZY:\n{context}\n\n"
        "Wybierz najbardziej trafny tekst z bazy do sytuacji ziomka z powyższego zdania. "
        "Wytnij z niego dwie linijki/wersy najbardziej odpowiadające sytuacji ziomka."
    )

    response = ollama.chat(
        model='SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M',
        messages=[
            {'role': 'system', 'content': system_instruction},
            {'role': 'user', 'content': prompt},
        ],
        options={
            'temperature': 0.1,
            'top_p': 0.9
        }
    )

    return response['message']['content']

if __name__ == "__main__":
    print("Elo! Jestem Twoim muzycznym wsparciem. Co u Ciebie?")
    while True:
        user_msg = input("Ty: ")
        if user_msg.lower() in ['exit', 'quit', 'pa']: break

        reply = lyrical_chat(user_msg)
        print(f"\nLyrical Homie: {reply}\n")