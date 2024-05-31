import os
import cohere
import pandas as pd
from sklearn.cluster import KMeans

k = 8
l = 100
cohere_API_key = "<<cohere api key>>"

def cluster_embeddings(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    cluster_labels = kmeans.labels_
    return cluster_labels

def main():
    # Embedding of texts
    co = cohere.Client(cohere_API_key)

    folder_path = "data/"
    file_list = os.listdir(folder_path)

    texts = []
    for file in file_list[:l]:
        print(file)
        with open(os.path.join(folder_path, file), 'r', encoding="utf8") as f:
            texts.append(f.read())

    response = co.embed(
      texts,
      model='embed-english-v3.0',
      input_type='clustering'
    )

    embeddings = response.embeddings

    # Clustering of embeddings
    cluster_labels = cluster_embeddings(embeddings, k)
    
    # Export results
    df = pd.DataFrame({'name': file_list[:l], 'text': texts,'cluster': cluster_labels})
    df.to_excel("results.xlsx")
    
if __name__ == "__main__":
    main()
