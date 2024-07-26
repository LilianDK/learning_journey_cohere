
import opensearch_py_ml as oml
import csv

from mockup_rag.config import (
    DATA_PATH,
    VECTOR_NAME,
    VECTOR_SIZE,
    client,
    COHERE_MODEL,
    INDEX_NAME,
)
from tqdm import tqdm
from mockup_rag.utils import get_cohere_embedding


def input_to_vector() -> list[dict[str, tuple[str, list[float]]]]:
    """Embedding of input to vector

    Returns:
        list[dict[str, tuple[str, list[float]]]]: List of dictionaries with the input and its embedded version
    """
    with open(DATA_PATH, newline="", encoding="Windows-1252") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=";")
        csv_input = [row for row in csv_reader if row["text"] != ""]

    # create list of texts
    texts = [row["text"] for row in csv_input]

    # embed all texts with cohere client
    embed_list = get_cohere_embedding(texts, model_name=COHERE_MODEL)

    # create a lookup table of text:vector
    for i, row in enumerate(csv_input):
        row[VECTOR_NAME] = embed_list[i]

    print("STATUS: TEXT EMBEDDING COMPLETED")
    return csv_input


# create index payload
body = {
    "settings": {"index": {"knn": "true", "knn.algo_param.ef_search": 100}},
    "mappings": {
        "properties": {
            VECTOR_NAME: {
                "type": "knn_vector",
                "dimension": VECTOR_SIZE,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {"ef_construction": 128, "m": 24},
                },
            },
        }
    },
}

if client.indices.exists(INDEX_NAME):
    client.indices.delete(INDEX_NAME)

client.indices.create(INDEX_NAME, body=body)
print(f"CREATED INDEX: {INDEX_NAME}")

# insert each row one-at-a-time to the document index
input = input_to_vector()

for i, row in tqdm(enumerate(input)):
    try:

        body = {
            VECTOR_NAME: row[VECTOR_NAME],
            "text": row["text"],
            "paragraph": row["paragraph"],
        }
        client.index(index=INDEX_NAME, id=i, body=body)
    except Exception as e:
        print(f"[ERROR]: {e}")
        continue

# sanity check inserted records
oml_df = oml.DataFrame(client, INDEX_NAME)
print(f"Shape of records inserted into index {INDEX_NAME} = {oml_df.shape}")
