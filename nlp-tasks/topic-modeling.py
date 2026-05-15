# Topic Modeling - automatically discover hidden topics in documents

# Eg topics - 
# topic - keywords
# sports - football, cricket, match
# finance - stocks, market, investment

# Common algo - LDA - Latent dirichlet allocation

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "football match goal team",
    "stock market investment finance",
    "soccer team championship",
    "trading stocks economy"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

lda = LatentDirichletAllocation(
    n_components=2,
    random_state=42
)

lda.fit(X)

print(lda.components_)