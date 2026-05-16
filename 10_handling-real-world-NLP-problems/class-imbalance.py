# Solution for Imbalance class - 
# - oversampling - increase minority samples
# - undersampling - reduce majority samples
# - class weights - penalize minority errors
# - f1-score - better evaluation

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    class_weight='balance'
)

# Interview Answer 
# How do you handle class imbalance?
# - use f1 score instead of accuracy
# - apply oversampling/undersampling
# - use class weights
# - collect better balanced data

