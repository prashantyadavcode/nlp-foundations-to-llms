# Chucking 

# Problem? - LLMs have context limits.
# cannot send - Entire PDF / entire databases

# Solution - split documents into chunks

# Example - 
# 100-page document
# ↓
# Smaller text chunks
# ↓
# Embedding generation
# ↓
# Vector storage

# Chunking Strategies - 
# Strategy - Description 
# fixed-size - equal token chunks
# recursive - split intelligently
# semnatic - meaning-aware chunking

text = """
Artificial intelligence is transforming industries.
Machine learning powers recommendation systems.
LLMs are changing software engineering.
"""

chunk_size = 50

chunks = [
    text[i:i+chunk_size]
    for i in range(0, len(text), chunk_size)
]

print(chunks)

# Why chunking matters - 
# Poor chunking causes - bad retrieval, hallucinations, missing context

