# F1 Score - most imp. classification metrics
#   = 2 * precision * recall / precision + recall
# Meaning - balances precision and recall
# why f1-score? important when - dataset is imbalanced, both FP and FN matters
# It balances precision and recall, making it more reliable than accuracy on imbalanced datasets

from sklearn.metrics import f1_score

y_true = [1, 1, 0, 0]
y_pred = [1, 0, 1, 0]

f1 = f1_score(
    y_true, y_pred
)

print(f1)