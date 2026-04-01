import os
import json
from pathlib import Path

INPUT_DIR = Path("lyrics")
processed_fingerprints = set()
unique_songs_count = 0
duplicates_count = 0


def get_song_fingerprint(title, lyrics):
    i = str(title or "").lower()
    j = str(lyrics or "").lower()[:50]

    clean_title = "".join([char for char in i if char.isalnum()])
    clean_lyrics = "".join([char for char in j if char.isalnum()])

    return f"{clean_title}_{clean_lyrics}"


files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]

for file_name in files:
    file_path = INPUT_DIR / file_name
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Structure: { "songs": [ {"title": "...", "lyrics": "..."}, ... ] }
    original_songs = data.get('songs', [])
    filtered_songs = []

    for song in original_songs:
        title = song.get('title', '')
        lyrics = song.get('lyrics', '')

        if not lyrics: continue

        fingerprint = get_song_fingerprint(title, lyrics)

        if fingerprint not in processed_fingerprints:
            processed_fingerprints.add(fingerprint)
            filtered_songs.append(song)
            unique_songs_count += 1
        else:
            duplicates_count += 1

    data['songs'] = filtered_songs
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Scan complete!")
print(f"Uniques: {unique_songs_count}")
print(f"Removed: {duplicates_count}")