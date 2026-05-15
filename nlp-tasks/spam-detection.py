from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

texts = [
    'Win money now',
    'Hello how are you',
    'Claim your free prize',
    "Let's meet tomorrow"
]

labels = [1, 0, 1, 0]

# TF-IDF
vectorizer = TfidfTransformer()

X = vectorizer.fit_transform(texts)

# Model 
model = LogisticRegression()
model.fit(X, labels)

# Prediction
test = vectorizer.transform(
    ['Free money offer']
)

prediction = model.predict(test)

print(prediction)

# similarly, for Toxicity detection - BERT, RoBERTa, DistilBERT commonly used

