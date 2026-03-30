import os
import json
from pathlib import Path

import lyricsgenius
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).parent


GENIUS_TOKEN = os.getenv("GENIUS_TOKEN")
OUTPUT_DIR = BASE_DIR / "lyrics"
ARTISTS_TO_DOWNLOAD = ["Oki", "Białas", "Żabson", "Pezet",
                       "Bedoes", "Zbuku", "Kaen", "Sentino",
                       "Malik-montana", "Diho", "Mata", "TEDE",
                        "Ostr", "Molesta", "Sokół", "Quebonafide",
                        "Taco Hemingway", "Ras", "Peja", "ReTo",
                       "Kaz Balagane", "Belmondo", "Fokus", "Donguralesko",
                       "Kizo", "Liroy", "Grubson", "Young-igi", "Eis",
                       "Hemp-gru", "Wwo", "Zip-skad", "Ten Typ Mes",
                       "Fisz", "Paktofonika", "Gruby-mielzky", "Dwa-sawy"]

genius = lyricsgenius.Genius(GENIUS_TOKEN)
genius.remove_section_headers = False
genius.skip_non_songs = True
genius.excluded_terms = ["(Remix)", "(Live)"]


def setup_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")


def download_artist_data(artist_name):
    print(f"\nLooking for: {artist_name}'s songs")

    artist = genius.search_artist(artist_name, max_songs=30, sort="popularity")

    if artist:
        safe_name = artist_name.replace(" ", "_")
        file_path = os.path.join(OUTPUT_DIR, f"{safe_name}_lyrics.json")

        artist_dict = artist.to_dict()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(artist_dict, f, ensure_ascii=False, indent=4)

        print(f"Success! {len(artist.songs)} songs of {artist_name} saved to: {file_path}")
    else:
        print(f"Error! Could not find artist: {artist_name}")


if __name__ == "__main__":
    setup_directory(OUTPUT_DIR)

    for artist in ARTISTS_TO_DOWNLOAD:
        try:
            download_artist_data(artist)
        except Exception as e:
            print(f"An error occurred while downloading {artist}: {e}")

    print("\n--- All tasks completed! ---")