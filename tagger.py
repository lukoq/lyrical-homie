import os
import json
import re
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv
from langdetect import detect_langs

load_dotenv()
client = genai.Client()

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "lyrics"
LYRICS_DIR = BASE_DIR / "lyrics"
GEMINI_DIR = LYRICS_DIR / "gemini"


GEMINI_DIR.mkdir(parents=True, exist_ok=True)

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

def clean_rap_lyrics(text):
    text = text.split('Lyrics', 1)[-1]
    text = text.rsplit('Embed', 1)[0]
    text = re.sub(r'\d+$', '', text)
    return text.strip()


def create_and_send_batch():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]

    metadata_registry = {}
    batch_requests = []
    request_counter = 0

    print("STEP 1: Text processing and package creation...")

    for file_name in files:
        with open(os.path.join(INPUT_DIR, file_name), 'r', encoding='utf-8') as f:
            data = json.load(f)

        artist = data.get('name', 'Unknown')

        for song in data.get('songs', []):
            title = song.get('title')
            lyrics = song.get('lyrics')
            if not lyrics: continue

            clean_text = clean_rap_lyrics(lyrics)
            chunks = split_rap_by_lines(clean_text)

            for i, chunk_data in enumerate(chunks):
                parent_text = chunk_data["parent"]
                child_text = chunk_data["child"]

                if not is_valid_polish(parent_text):
                    continue

                chunk_id = f"{artist}_{title}_{i}".replace(" ", "_").replace("/", "_").lower()

                metadata_registry[chunk_id] = {
                    "artist": artist,
                    "title": title,
                    "child": child_text,
                    "parent": parent_text
                }

                prompt = (
                    "Jesteś analitykiem. Wyciągnij do 5 słów kluczowych (oficjalnych pojęć słownikowych) z poniższego tekstu rapu. "
                    "MUSISZ ZWRÓCIĆ DOKŁADNIE TAKI FORMAT JSON:\n"
                    f'{{"id": "{chunk_id}", "tags": ["tag1", "tag2"]}}\n\n'
                    f"Tekst:\n{parent_text}"
                )

                request_obj = {
                    "request": {
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "temperature": 0.1
                        }
                    }
                }
                batch_requests.append(request_obj)
                request_counter += 1

    print(f"\nQueries have been created for {request_counter} verses.")

    registry_path = LYRICS_DIR / "local_metadata_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(metadata_registry, f, ensure_ascii=False, indent=2)
    print(f" -> Saved '{registry_path}'")

    batch_filename = GEMINI_DIR / "gemini_batch_requests.jsonl"
    with open(batch_filename, "w", encoding="utf-8") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
    print(f" -> The batch file has been saved '{batch_filename}'")

    print("\nSTEP 2: Uploading the file to Google's servers...")
    uploaded_file = client.files.upload(
        file=str(batch_filename),
        config=types.UploadFileConfig(mime_type="text/plain")
    )
    print(f" -> File uploaded! URI: {uploaded_file.uri}")

    print("\nSTEP 3: Launch Batch API...")
    batch_job = client.batches.create(
        model="gemini-2.5-flash",
        src=uploaded_file.name
    )

    print("==================================================")
    print(f">>> {batch_job.name} <<<")
    print("==================================================")


if __name__ == "__main__":
    create_and_send_batch()