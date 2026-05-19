# Fine Tuning
# - adpat model to specific domain/task

# Eg - medical chatbot trained further on: Clinical reports, healthcare conversations.

# Type of Fine-Tuning - 
# Type - Meaning
# Full-Fine-Tuning - Train all parameters
# LoRA - Train-lightweight adapaters
# QLoRA - Quantized LoRA

# LoRA Eg using Hugging Face PEFT

from peft import LoraConfig

config = LoraConfig(
    r = 8,
    lora_alpha = 16,
    target_modules = ['q_proj', 'v_proj']
)

# Why LoRA is important ->
# Benefits -> lower GPU memory, Faster training, cheap fine-tuning

