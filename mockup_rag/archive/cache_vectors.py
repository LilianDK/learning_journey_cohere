import json

import pandas as pd
from mockup_rag.config import COHERE_MODEL, DATA_PATH, EMBED_COLUMN
from mockup_rag.utils import get_cohere_embedding

df = pd.read_csv(DATA_PATH, 
                 encoding="Windows-1252", 
                 delimiter=";"
                 ).fillna("").reset_index(drop=True)

# create list of texts
texts = []
for text in df[EMBED_COLUMN].values.tolist():
    texts.append(text[0])

# embed all texts with cohere client
embed_list = get_cohere_embedding(texts, model_name=COHERE_MODEL)

# create a lookup table of text:vector
cache = dict(zip(texts, embed_list))

# dump out cache
with open("cache.jsonl", "w") as fp:
    json.dump(cache, fp)

print("STATUS: Created cache.jsonl")