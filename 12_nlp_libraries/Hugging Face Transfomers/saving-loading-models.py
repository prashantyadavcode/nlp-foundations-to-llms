# Save Model 

model.save_pretrained("./my_model")

tokenizer.save_pretrained("./my_model")


# Load Model 

from transformers import AutoModel

model = AutoModel.from_pretrained(
    "./my_model"
)