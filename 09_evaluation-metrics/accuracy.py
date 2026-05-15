# Accuracy - TP + TN / TP + TN + FP + FN
# precentage of correct prediction

from sklearn.metrics import accuracy_score

y_true = [1, 1, 0, 0]
y_pred = [1, 0, 1, 0]

acc = accuracy_score(y_true, y_pred)
print(acc)

# works worse on imbalanced datasets

