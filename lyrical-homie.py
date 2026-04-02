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


def get_lyrics_context(user_input, hypothetical_rap):
    results_hyde = collection.query(
        query_texts=[hypothetical_rap],
        n_results=30
    )
    results_user = collection.query(
        query_texts=[user_input],
        n_results=30
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
        clean_doc = doc.replace("passage: ", "").strip()
        context += f"[{meta['artist']} - {meta['title']}]\n{clean_doc}\n---\n"
        context += f"Wynik: {scores[idx]:.4f} \n\n"

    return context




def lyrical_chat(user_input):
    hyde_system_instructions = (
        "Jesteś systemem, który zamienia zdanie użytkownika na cztery uliczne wersy polskiego rapu.\n"
        "ZASADA 1: Napisz DOKŁADNIE CZTERY wersy.\n"
        "ZASADA 2: Nie dodawaj absolutnie żadnych wstępów.\n"
        "ZASADA 3: Nie numeruj wersów.\n"
        "ZASADA 4: Od razu zacznij pisać rymy."
    )

    hyde_response = ollama.chat(
        model='SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M',
        messages=[
            {'role': 'system', 'content': hyde_system_instructions},
            {'role': 'user', 'content': user_input},
        ],
        options={
            'temperature': 0.5,
            'stop': ['<|im_id|>', '<|im_end|>', '<|im_sep|>']
        }
    )

    hypothetical_rap = hyde_response['message']['content'].strip()
    context = get_lyrics_context(user_input, hypothetical_rap)

    print(hypothetical_rap)
    print(context)

    system_instruction = (
        "Jesteś 'Lyrical Homie'. Odpowiadasz WYŁĄCZNIE cytatami z polskiego rapu.\n\n"
        "INSTRUKCJA:\n"
        "1. Otrzymasz poniżej kilka cytatów z polskiego rapu i 1 opis sytuacji ziomka.\n"
        "2. Wybierz TYLKO JEDEN cytat, który najlepiej odnosi się do opisanej sytuacji.\n"
        "3. Upewnij się, że wybrany cytat stanowi logiczną ODPOWIEDŹ na problem ziomka.\n"
        "4. IGNORUJ metadane w nawiasach kwadratowych (np. autor i tytuł)"
        "5. NIE PISZ absolutnie nic więcej. Żadnych wstępów, pozdrowień ani własnych komentarzy."
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