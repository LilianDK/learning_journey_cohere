import cohere
from fastapi import FastAPI

from mockup_rag.config import COHERE_API_KEY, INDEX_NAME
from opensearchpy import OpenSearch

INDEX_NAME = "asyl-cosine"

app = FastAPI()

cohere_client = cohere.Client(COHERE_API_KEY)


@app.get("/")
def index():
    return {"message": "Make a post request to /search to search through news articles"}


@app.post("/search")
def search(query: str):
    query_embedding = cohere_client.embed(texts=[query], model="small").embeddings[0]

    similar_news = client.search(
        index=INDEX_NAME,
        body={
            "query": {"knn": {"embedding": {"vector": query_embedding, "k": 10}}},
        },
    )
    response = [
        {
            "paragraph": r["_source"]["paragraph"],
            "text": r["_source"]["text"],
        }
        for r in similar_news["hits"]["hits"]
    ]

    return {
        "response": response,
    }