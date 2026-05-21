# Clustering - embeddings enable grouping by meaning
# Eg - 

from sklearn.cluster import KMeans

sentences = [
    'I love football',
    'Soccer is amazing',
    'AI is transforming industries',
    'Machine Learning is powerful'
]

embeddings = model.encode(sentences)

kmeans = KMeans(
    n_clusters = 2,
    random_state = 42
)

kmeans.fit(embeddings)
print(kmeans.labels_)

# Output - [0, 0, 1, 1] - sports grouped together and AI grouped together.
