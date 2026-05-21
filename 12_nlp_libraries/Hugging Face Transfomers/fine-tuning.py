# Fine Tuning Workflow - 

# Pretrained Model
# ↓
# Custom Dataset
# ↓
# Training
# ↓
# Task-Specific Model

from transformers import Trainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8
)

trainer = Trainer(
    model=model,
    args=training_args
)

