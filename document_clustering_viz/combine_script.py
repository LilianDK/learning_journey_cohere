import pandas as pd
import json
import polars as pl
import ollama
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
from kneed import KneeLocator

df_type_statistics = []
df_type_statistics_count = []
peeps = []

def cluster_embeddings(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    cluster_labels = kmeans.predict(embeddings)
    #cluster_labels = kmeans.labels_
    return cluster_labels

# ------------------------------------------------------------- EMBEDDING 
# Embedding of texts
embeddings = []
model = 'mxbai-embed-large'

for prompt in peeps:
    e = ollama.embeddings(
        model=model ,
        prompt=prompt,
        )
    embeddings.append(e["embedding"])

# ------------------------------------------------------------- IDENTIFY K
# Initialize kmeans parameters
kmeans_kwargs = {
"init": "random",
"n_init": 100,
"random_state": 1,
}

# Calculate SSE for given k range
sse = []
start = 1
end = 20
x = np.arange(start, end)

for k in range(start, end):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(embeddings)
    sse.append(kmeans.inertia_)

# Visualize k range
kneedle = KneeLocator(x=x, y=sse, S=1.0, curve="convex", direction="decreasing")
print(kneedle.knee)
print(kneedle.elbow)
print(round(kneedle.knee_y, 3))
print(f"The optimal k is k={kneedle.knee}")
kneedle_plot1 = kneedle.plot_knee_normalized()
kneedle_plot2 = kneedle.plot_knee()

#kneedle_plot1.savefig(f"results/images/{model}_kneedle_plot1.png", dpi=300) 
#kneedle_plot2.savefig(f"results/images/{model}_kneedle_plot2.png", dpi=300) 
# ------------------------------------------------------------- CLUSTERING
# Clustering of embeddings
k = kneedle.knee
cluster_labels = cluster_embeddings(embeddings, k)

# Export results
df = pd.DataFrame({'name': peeps, 'cluster': cluster_labels})
df.to_excel(f"results/{model}_results.xlsx")
df.head()

# ------------------------------------------------------------- Visualization 1
# Colors
colors = ["#9E292B", "#54616E", "#81BED7", "#39756F", "#7030A0", "#9FA4A9", "#FFC000", "#48A689", "#2B5DC1", "#B3D8E7"]
palette = colors[:k]
print(palette)

# Instantiation of tsne, specify cosine metric
tsne = TSNE(random_state = 0, n_iter = 1000, metric = 'cosine')
# Fit and transform
embeddings = np.array(embeddings)
embeddings2d = tsne.fit_transform(embeddings)

embeddingsdf = pd.DataFrame()
embeddingsdf['cluster'] = df.cluster
embeddingsdf['x'] = embeddings2d[:,0]
embeddingsdf['y'] = embeddings2d[:,1]
df["2d_x"] = embeddingsdf['x']
df["2d_y"] = embeddingsdf['y']
embeddingsdf.head()

df.to_excel(f"results/{model}_2d_transformation.xlsx")

img_cluster = sns.lmplot(data=embeddingsdf, x='x', y='y', hue='cluster', palette=palette,
                            scatter_kws={'alpha':0.5}, 
                            fit_reg=False, legend=True, legend_out=True)
ax = plt.gca()
ax.set_title(f"Scatterplot with t-sne, k={k}, model={model}")

img_cluster.savefig(f"results/images/{model}_cluster_overview.png", dpi=300) 

# ------------------------------------------------------------- Visualization 2
for i in range(0, k):
    cluster = i
    subset = embeddingsdf[embeddingsdf["cluster"] == cluster]
    sub_palette = colors[cluster:cluster+1]

    img_sub_cluster = sns.lmplot(data=subset, x='x', y='y', hue='cluster', palette=sub_palette,
                            scatter_kws={'alpha':0.5}, 
                            fit_reg=False, legend=True, legend_out=True)
    ax = plt.gca()
    ax.set_title(f"Scatterplot with t-sne, k={k}, model={model}, cluster={cluster}")

    img_sub_cluster.savefig(f"results/images/sub_images/{model}_cluster_overview_cluster_{i}.png", dpi=300) 


