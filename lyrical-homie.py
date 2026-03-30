from pathlib import Path
import chromadb
import ollama


BASE_DIR = Path(__file__).parent


CHROMA_PATH = BASE_DIR / "vector_db"
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name="polish_rap_lyrics")



def get_lyrics_context(user_query):
    results = collection.query(
        query_texts=[user_query],
        n_results=1
    )

    context = ""
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        context += f"[{meta['artist']} - {meta['title']}]: {doc}\n---\n"
    return context



def lyrical_chat(user_input):
    context = get_lyrics_context(user_input)
    system_instruction = (
        "Jesteś 'Lyrical Homie' – kumplem z osiedla, który na każdą sytuację i problem "
        "użytkownika odpowiada wyłącznie trafnie dobranym cytatem z polskiego rapu.\n\n"
        "TWOJE ZASADY:\n"
        "1. ZERO LANIA WODY: Bądź krótki, konkretny i luźny. Żadnych moralitetów i długich porad.\n"
        "2. TYLKO FAKTY: Wybieraj cytaty TYLKO z dostarczonego kontekstu. Nigdy nie zmyślaj własnych rymów.\n"
        "3. STYL: Skomentuj sytuację jednym, krótkim, ziomalskim cytatem. \n"
    )

    prompt = (
        f"SYTUACJA ZIOMKA: {user_input}\n\n"
        f"DOSTĘPNE WERSY Z BAZY:\n{context}\n\n"
        "Wybierz najbardziej trafny tekst do sytuacji z powyższego zdania i rzuć odpowiedzią w stylu Lyrical Homie:"
    )

    response = ollama.chat(
        model='SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M',
        messages=[
            {'role': 'system', 'content': system_instruction},
            {'role': 'user', 'content': prompt},
        ],
    )

    return response['message']['content']

# 3. Pętla czatu
if __name__ == "__main__":
    print("Elo! Jestem Twoim muzycznym wsparciem. Co u Ciebie?")
    while True:
        user_msg = input("Ty: ")
        if user_msg.lower() in ['exit', 'quit', 'pa']: break

        reply = lyrical_chat(user_msg)
        print(f"\nLyrical Homie: {reply}\n")