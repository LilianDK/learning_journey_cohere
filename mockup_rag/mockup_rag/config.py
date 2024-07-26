import os

import cohere
from dotenv import load_dotenv
from opensearchpy import OpenSearch

load_dotenv("../.env")

COHERE_API_KEY = os.environ["COHERE_API_KEY"]
COHERE_MODEL = "embed-multilingual-v2.0"
VECTOR_NAME = "cohere_vector"
VECTOR_SIZE = "768"
DATA_PATH = "../data/response_discoveryv2_asylgesetz_paragraph_split.csv"
INDEX_NAME = "test2-cosine"

# init your clients
co = cohere.Client(COHERE_API_KEY)
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True,  # enables gzip compression for request bodies
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)
