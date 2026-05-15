# Evaluation Metrics

# Classification Metrics - 
# Metric - Meaning
# Accuracy - correct prediction
# Precision - correct positive predictions
# Recall - coverage of positives
# F1-score - balance of precision/recall

# NER Metrics
# - Precision
# - Recall
# - F1 score

# Translation Metrics
# Metric - Use
# BLEU - Translation quality
# ROUGE - summarization quality

# Semantic Search Metrics
# Metric - Meaning
# Cosine Similarity - Embedding similarity
# Recall@K - relevant retrieval quality

from sklearn.metrics import classification_report

y_true = [1, 0, 1, 1]
y_pred = [1, 0, 0, 1]

print(classification_report(y_true, y_pred))


