# Precision - measures prediction quality
#  = TP / TP + FP
# actual positives out of predicted positives

# real life eg - spam detection : high precision means - few legitimate emails marked as spam

from sklearn.metrics import precision_score

y_true = [1, 1, 0, 0]
y_pred = [1, 0, 1, 0]

precision = precision_score(
    y_true, y_pred
)

print(precision)