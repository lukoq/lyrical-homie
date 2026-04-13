import json
import time
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "lyrics"
REGISTRY_PATH = INPUT_DIR / "local_metadata_registry.json"

# --- Config ---
MODEL_NAME = "gemini-2.5-flash-lite"
DELAY = 0.2
LIMIT = 1000


def get_tags_gemini(text_chunk):
    prompt = (
        "Jesteś ekspertem od polskiego rapu. Wyciągnij do 5 słów kluczowych. "
        "ZAMIEŃ SLANG NA POJĘCIA OFICJALNE (np. 'kapusta' -> 'pieniądze'). "
        "Używaj małych liter i mianownika. "
        f"Zwróć WYŁĄCZNIE JSON: {{\"tags\": [\"tag1\", \"tag2\"]}}\n\n"
        f"Tekst: {text_chunk}"
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        data = json.loads(response.text)
        return [t.lower().strip() for t in data.get("tags", [])]
    except Exception as e:
        print(f"\n[!] Error: {e}")
        return None


def run_tagger():
    if not REGISTRY_PATH.exists():
        print("[!] File does not exist.")
        return

    with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
        registry = json.load(f)

    to_process = [cid for cid, val in registry.items() if not val.get("tags")]

    to_process = to_process[:LIMIT]
    total = len(to_process)

    if total == 0:
        print("[*] No more records to tagged")
        return

    print(f"[*] LAUNCHING TEST: {total} records...")

    processed_count = 0
    for idx, chunk_id in enumerate(to_process, 1):
        parent_text = registry[chunk_id]["parent"]

        tags = get_tags_gemini(parent_text)

        if tags:
            registry[chunk_id]["tags"] = tags
            print(f"[{idx}/{total}] {chunk_id} -> {tags}")
            processed_count += 1

        if idx % 50 == 0:
            with open(REGISTRY_PATH, 'w', encoding='utf-8') as f:
                json.dump(registry, f, ensure_ascii=False, indent=2)
            print(f">>> Saved: {idx}/{total}")

        time.sleep(DELAY)

    with open(REGISTRY_PATH, 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

    print(f"\nTagged {processed_count} records.")



if __name__ == "__main__":
    run_tagger()