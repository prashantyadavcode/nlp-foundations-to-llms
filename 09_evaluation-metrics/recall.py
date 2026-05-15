# Recall - measures coverage
#  = TP / TP + FN

# out of actual positives , how many did we find?

# Eg - high recall important because - missing a positive case is dangerous

from sklearn.metrics import recall_score

y_true = [1, 1, 0, 0]
y_pred = [1, 0, 1, 0]

recall = recall_score(
    y_true, y_pred
)

