# Named Entity Recognition (NER)

# What is NER? Identify entities in text 
# Eg - Sentence - 'Elon Musk founded tesla in california'
#    - NER Output - 'Elon Musk' - PERSON, 'Tesla' - ORG, 'california' - LOCATION

# Common entity types - 
# Entity - ORG - Organizations, LOC/GPE - Locations, DATE - Dates, MONEY - Curreny values

import spacy

nlp = spacy.load('en_core_web_sm')
text = 'Elon musk founded Tesla in California'
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)

# Output - 

# Elon Musk PERSON
# Tesla ORG
# California GPE


