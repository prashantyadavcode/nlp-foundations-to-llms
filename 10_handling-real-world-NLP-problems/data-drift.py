# What is data drift?
# - production data distribution changes over time
# Eg - spam model trained in 2022: 'Win lottery now', Modern spam: 'Claim crypto rewards' 
# Patterns changed -> model performance drop

# Types of drift
# drift - meaning
# data drift - input distribution changes
# concept drift - meaning changes
# label drift - output distribution changes

# Monitoring drift
# method - purpose
# KL divergence - distribution comparison
# PSI - population statibility
# Embeddding monitoring - semantic drift
# performance metrics - Accuracy/F1 drops

import numpy as np
from scipy.stats import ks_2samp

train_data = np.random.normal(0, 1, 1000)
prod_data = np.random.normal(1, 1, 1000)

stat, p_value = ks_2samp(
    train_data, prod_data
)

print(p_value) # low p-value may indicate drift

# How would you monitor drift? (interview)
# - monitor input distribution
# - track prediction confidence
# - compare embedding/statistics
# - continuously evaluate production metrics
# - retrain periodically


