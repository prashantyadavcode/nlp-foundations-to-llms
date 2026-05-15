# ROC_AUC - measures ranking ability

# ROC Curve - Plots -
# axis - meaning
# x - false postive rate
# y - true positive rate

# AUC - area under curve
# higher auc - better class separation

from sklearn.metrics import roc_auc_score

y_true = [0, 1, 1, 0]
y_scores = [0.1, 0.8, 0.7, 0.2]

auc = roc_auc_score(
    y_true, y_scores
)

print(auc)

