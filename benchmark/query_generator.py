import json
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm
import os

from config.prompts import BENCHMARK_QUERY_PROMPT

BASE_DIR = Path(__file__).parent
SOURCE_FILE = BASE_DIR.parent / "lyrics" / "local_metadata_registry.json"
OUTPUT_FILE = BASE_DIR / "datasets" / "benchmark_dataset_sample.json"

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

MODEL_QUERY = "llama-3.3-70b-versatile"
SAMPLE_SIZE = 50

def get_natural_query(child_text, artist):
    """Llama 70B"""

    prompt = BENCHMARK_QUERY_PROMPT.format(
        artist=artist,
        child_text=child_text
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], # type: ignore
            model=MODEL_QUERY,
            temperature=0.9
        )
        return chat_completion.choices[0].message.content.strip().replace('"', '').lower()
    except Exception as e:
        print(f"Groq ERROR: {e}")
        return "Erro"


def main():
    if not SOURCE_FILE.exists():
        print(f"File doesn't exist: {SOURCE_FILE}")
        return

    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_keys = list(data.keys())
    sampled_keys = random.sample(all_keys, min(SAMPLE_SIZE, len(all_keys)))
    benchmark_data = []

    print(f"Creating {len(sampled_keys)} queries by Llama 3.3 70B...")

    for key in tqdm(sampled_keys):
        item = data[key]

        query = get_natural_query(item['child'], item['artist'])

        benchmark_data.append({
            "test_id": key,
            "query": query,
            "expected_intent": "",
            "expected_child": item['child'],
            "expected_artist": item['artist']
        })

        time.sleep(2.1)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)

    print(f"Ready! File name: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()