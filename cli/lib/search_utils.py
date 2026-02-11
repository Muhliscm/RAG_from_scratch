import json
from pathlib import Path
from shutil import which


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT/'data'
MOVIES_PATH = DATA_PATH/'movies.json'
STOP_WORDS_PATH = DATA_PATH/'stopwords.txt'

CACHE_PATH = PROJECT_ROOT/'cache'

def load_movies() -> list[dict]:
    with open(MOVIES_PATH,'r') as f:
        data = json.load(f)
    return data['movies']

def load_stop_words():
    with open(STOP_WORDS_PATH,'r') as f:
        data = f.readlines()
        data = [word.strip() for word in data]
    return data