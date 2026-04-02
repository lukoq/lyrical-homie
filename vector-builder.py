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
    model_name="sdadas/mmlw-e5-base"
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


def split_rap_by_lines(text, chunk_lines=4, overlap_lines=2):
    blocks = re.split(r'\n\s*\n', text.strip())

    chunks = []
    step = chunk_lines - overlap_lines

    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]

        if not lines:
            continue

        if lines[0].startswith('[') and lines[0].endswith(']'):
            lines = lines[1:]

        if not lines:
            continue

        if len(lines) <= chunk_lines:
            chunks.append("\n".join(lines))
            continue

        for i in range(0, len(lines), step):
            chunk_group = lines[i:i + chunk_lines]
            chunk_text = "\n".join(chunk_group)
            chunks.append(chunk_text)

            if i + chunk_lines >= len(lines):
                break

    return chunks



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

            if not lyrics: continue

            clean_text = clean_rap_lyrics(lyrics)
            chunks = split_rap_by_lines(clean_text, chunk_lines=4, overlap_lines=2)

            for i, chunk in enumerate(chunks):

                if not is_valid_polish(chunk): # Reject non polish verses
                    print(chunk)
                    continue


                vector_content = f"passage: {chunk}"
                safe_id = f"{artist}_{title}_{i}".replace(" ", "_").replace("/", "_").lower()

                collection.add(
                    documents=[vector_content],
                    metadatas=[{
                        "artist": artist,
                        "title": title,
                        "source": "Genius",
                        "chunk_id": i
                    }],
                    ids=[safe_id]
                )
if __name__ == "__main__":
    process_lyrics()
    print("Vector Database created with Polish context!")