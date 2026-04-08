import os
import json
import re
from pathlib import Path

import ollama
from huggingface_hub import login
from dotenv import load_dotenv
from langdetect import detect_langs

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "lyrics"
DATABASE_PATH = BASE_DIR / "vector_db"
REGISTRY_PATH = INPUT_DIR / "local_metadata_registry.json"
ARTISTS_DIR = INPUT_DIR / "artists"


polish_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sdadas/mmlw-e5-base",
    device="cpu"
)

client = chromadb.PersistentClient(path=str(DATABASE_PATH))

try:
    client.delete_collection(name="polish_rap_lyrics")
except:
    pass

collection = client.create_collection(
    name="polish_rap_lyrics",
    embedding_function=polish_ef,
    metadata={"hnsw:space": "cosine"}
)


def generate_metadata_tags(text): # For local use
    prompt = (
        "Jesteś analitykiem. Twoim jedynym zadaniem jest wyciągnięcie maksymalnie 5 słów kluczowych "
        "z poniższego tekstu. Używaj oficjalnych, słownikowych pojęć. Zwróć plik JSON z jednym kluczem 'tags'.\n\n"
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

        tags_list = data.get('tags', [])
        raw_string = ", ".join(tags_list).lower()

        if "tagi:" in raw_string:
            raw_string = raw_string.split("tagi:")[-1]

        if "zwrócone" in raw_string:
            raw_string = raw_string.replace("zwrócone", "")

        bad_chars = ["[", "]", "tekst:", "słowa kluczowe:"]
        for char in bad_chars:
            raw_string = raw_string.replace(char, "")

        return raw_string.strip()

    except Exception as e:
        print(f"   [Błąd LLM: {e}]")
        return ""

def clean_rap_lyrics(text):
    text = text.split('Lyrics', 1)[-1]
    text = text.rsplit('Embed', 1)[0]
    text = re.sub(r'\d+$', '', text)
    return text.strip()


# Splitting verse by lines as a child and parent (Parent-Document Retrieval)
def split_rap_by_lines(text, child_lines=2, parent_context_lines=2):
    blocks = re.split(r'\n\s*\n', text.strip())
    chunks_data = []

    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]

        if lines and lines[0].startswith('[') and lines[0].endswith(']'):
            lines = lines[1:]

        if not lines:
            continue

        for i in range(0, len(lines), child_lines):
            child_group = lines[i:i + child_lines]
            child_text = "\n".join(child_group)

            if not child_text.strip():
                continue

            start_idx = max(0, i - parent_context_lines)
            end_idx = min(len(lines), i + child_lines + parent_context_lines)

            parent_group = lines[start_idx:end_idx]
            parent_text = "\n".join(parent_group)

            chunks_data.append({
                "child": child_text,
                "parent": parent_text
            })

    return chunks_data


def is_valid_polish(chunk):
    try:
        langs = detect_langs(chunk)

        for lang in langs:
            if lang.lang == 'pl' and lang.prob > 0.4:
                return True
        return False
    except:
        return False


def build_metadata_registry(): # Use this script before call process_lyrics_with_tags() to build local_metadata_registry.json file

    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    registry = {}

    files = [f for f in os.listdir(ARTISTS_DIR) if f.endswith('.json') and f != "local_metadata_registry.json"]

    if not files:
        print("Does not found any lyrics in 'lyrics/artists/ directory'.")
        return

    processed_songs = 0
    total_chunks = 0

    for file_name in files:
        file_path = INPUT_DIR / file_name

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Decode error: {file_name}")
                continue

        artist = data.get('name', 'Unknown')
        songs = data.get('songs', [])

        for song in songs:
            title = song.get('title')
            lyrics = song.get('lyrics')

            if not lyrics or not title:
                continue

            processed_songs += 1

            clean_text = clean_rap_lyrics(lyrics)
            chunks = split_rap_by_lines(clean_text)

            for i, chunk_data in enumerate(chunks):
                child_text = chunk_data["child"]
                parent_text = chunk_data["parent"]

                if not is_valid_polish(parent_text):
                    continue

                raw_id = f"{artist}_{title}_{i}"
                safe_id = re.sub(r'[^a-z0-9]', '_', raw_id.lower())

                safe_id = re.sub(r'_+', '_', safe_id).strip('_')

                registry[safe_id] = {
                    "artist": artist,
                    "title": title,
                    "child": child_text,
                    "parent": parent_text,
                    "tags": []
                }
                total_chunks += 1

        print(f"[+] {artist} ({len(songs)} records)")


def process_lyrics(): # OLD METHOD
    files = [f for f in os.listdir(ARTISTS_DIR) if f.endswith('.json')]

    for file_name in files:
        file_path = ARTISTS_DIR / file_name
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        artist = data.get('name', 'Unknown')
        songs = data.get('songs', [])

        print(f"Processing {len(songs)} songs from {artist}...")

        for song in songs:
            title = song.get('title')
            lyrics = song.get('lyrics')

            if not lyrics:
                continue

            clean_text = clean_rap_lyrics(lyrics)
            chunks = split_rap_by_lines(clean_text)

            docs_batch = []
            metas_batch = []
            ids_batch = []

            for i, data in enumerate(chunks):
                child_text = data["child"]
                parent_text = data["parent"]

                if not is_valid_polish(parent_text): # Reject non polish verses
                    continue

                tags = generate_metadata_tags(parent_text)

                if tags:
                    enriched_child = f"passage: [Tagi: {tags}] {child_text}"
                    enriched_parent = f"[Kontekst: {tags}]\n{parent_text}"
                else:
                    enriched_child = f"passage: {child_text}"
                    enriched_parent = parent_text

                docs_batch.append(f"passage: {enriched_child}")
                metas_batch.append({
                    "artist": artist,
                    "title": title,
                    "chunk_id": i,
                    "parent_text": enriched_parent
                })
                safe_id = f"{artist}_{title}_{i}".replace(" ", "_").replace("/", "_").lower()
                ids_batch.append(safe_id)

            if docs_batch:
                collection.add(
                    documents=docs_batch,
                    metadatas=metas_batch,
                    ids=ids_batch
                )


def process_lyrics_with_tags(): # NEW METHOD
    if not REGISTRY_PATH.exists():
        print("local_metadata_registry.json has not found!")
        return

    with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
        registry = json.load(f)

    total_records = len(registry)
    print(f"Found {total_records} records.")

    docs_batch = []
    metas_batch = []
    ids_batch = []

    BATCH_SIZE = 2000
    processed = 0

    for chunk_id, data in registry.items():
        child_text = data["child"]
        parent_text = data["parent"]
        tags = data.get("tags", [])

        if tags:
            tags_str = ", ".join(tags)
            doc_text = f"passage: [Tagi: {tags_str}] {child_text}"
            parent_meta = f"[Kontekst: {tags_str}]\n{parent_text}"
        else:
            tags_str = ""
            doc_text = f"passage: {child_text}"
            parent_meta = parent_text

        docs_batch.append(doc_text)

        metas_batch.append({
            "artist": data["artist"],
            "title": data["title"],
            "parent_text": parent_meta,
            "tags": tags_str
        })
        ids_batch.append(chunk_id)

        processed += 1

        if len(docs_batch) >= BATCH_SIZE:
            collection.add(
                documents=docs_batch,
                metadatas=metas_batch,
                ids=ids_batch
            )
            print(f"Batch saved: {processed}/{total_records}")

            docs_batch.clear()
            metas_batch.clear()
            ids_batch.clear()

    if docs_batch:
        collection.add(
            documents=docs_batch,
            metadatas=metas_batch,
            ids=ids_batch
        )
        print(f"Saved: {processed}/{total_records}")


if __name__ == "__main__":
    process_lyrics_with_tags()
    print("Vector Database has been created with Polish context!")