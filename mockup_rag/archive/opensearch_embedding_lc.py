from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

path = "data/response_discoveryv2_asylgesetz_paragraph_split.csv"
loader = CSVLoader(file_path=path, source_column="text", csv_args={"delimiter": ";"})

embeddings = CohereEmbeddings(
    model="embed-multilingual-v2.0"
)

docs = loader.load()

docsearch = OpenSearchVectorSearch.from_documents(
    docs, embeddings, opensearch_url="http://localhost:9200",      
    #http_auth=("admin", "Tef2magi9!"),
    use_ssl = True,
    verify_certs = False,
    ssl_assert_hostname = False,
    ssl_show_warn = False,
)

query = "wer ist asylberechtigt"
docs = docsearch.similarity_search(query, k=10)

print(docs[0].page_content)

import chardet

def detect_encoding(path):
    with open(path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        print(f"Detected encoding: {encoding} with confidence {confidence}")
    return encoding

# Beispielverwendung
file_path = path
encoding = detect_encoding(file_path)

# Datei mit dem ermittelten Encoding lesen
with open(file_path, 'r', encoding=encoding) as file:
    content = file.read()
    print(content)