# Classification Metrics

# used in - spam detection, sentimental analysis, toxicity detection, intent Classification
# eg - 
# actual - predicted
# spam - spam
# spam - not spam
# not spam - spam
# not spam - not spam


# Confusion Matrix - foundation of Classification Metrics

# TP - actual positive, predicted positive
# FN - actual postive, predicted negative
# FP - actual negative, predicted positive
# TN - actual negative, predicted negative

from sklearn.metrics import confusion_matrix

y_true = [1, 1, 0, 0],
y_pred = [1, 0, 1, 0]

cm = confusion_matrix(y_true, y_pred)
print(cm)