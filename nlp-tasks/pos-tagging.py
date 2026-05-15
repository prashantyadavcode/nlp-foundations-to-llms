# POS Tagging (part-of-speech tagging)

# assign grammatical tags
# Eg - sentence - 'The dog runs fast'
# Common POS Tags - 
# The - Determiner
# dog - noun
# runs - verb
# fast - adverb

# common post tags - 
# NN - noun
# VB - verb
# JJ - adjective
# RB - adverb

import spacy 

nlp = spacy.load('en_core_web_sm')
text = 'The dog runs fast'
doc = nlp(text)

for token in doc:
    print(token.text, token.pos_)


# Output 
# The DET
# dog NOUN
# runs VERB
# fast ADV



