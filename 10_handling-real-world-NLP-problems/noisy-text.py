# Noisy Text - Real-world text is messy

# Eg - heyyyyy!!!, gr8ttt, LOL 😂😂😂
# Challenges - Typos, Emojis, Slang, Abbreviations, Mixed languages

# Solutions - 
# Method - Purpose
# text normalization - standardize
# spell correction - fix typos
# emoji handling - preserve meaning
# robust tokenizers - better parsing


import re 

text = 'heyyyy!!! I looooove NLP 😂'
text = re.sub(r'(.)\+1', r'\1', text)
print(text)

