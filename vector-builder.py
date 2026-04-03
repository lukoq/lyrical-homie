import os
import json
import re
from pathlib import Path
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


def process_lyrics():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]

    for file_name in files:
        file_path = INPUT_DIR / file_name
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

                docs_batch.append(f"passage: {child_text}")
                metas_batch.append({
                    "artist": artist,
                    "title": title,
                    "chunk_id": i,
                    "parent_text": parent_text
                })
                safe_id = f"{artist}_{title}_{i}".replace(" ", "_").replace("/", "_").lower()
                ids_batch.append(safe_id)

            if docs_batch:
                collection.add(
                    documents=docs_batch,
                    metadatas=metas_batch,
                    ids=ids_batch
                )


if __name__ == "__main__":
    process_lyrics()
    print("Vector Database has been created with Polish context!")