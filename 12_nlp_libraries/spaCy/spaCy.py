# spaCy is one of the most important NLP libraries for production systems.
    
# Used heavily for - 
#     - Information execution
#     - NER pipelines
#     - Text preprocessing
#     - Search systems
#     - Chatbots 
#     - Enterprise NLP workflows

# Unlike NLTK: spaCy is designed for production and speed

# Installation - pip3 install spacy
# Download English model - python -m spacy download en_core_web_sm

# Loading spaCy model

import spacy
nlp = spacy.load('en_core_web_sm')
text = 'Apple hired John in California'
doc = nlp(text)

print(doc)


# Tokenization 

doc = nlp('I love NLP and Transformers!')

for token in doc:
    print(token.text)


# Useful Token Attributes
# Attributes - Meaning
# token.text - original text
# token.lower_ - lowercase
# token.is_stop - stopwords
# token.is_punct - punctuation

for token in doc:
    print(
        token.text,
        token.lower_,
        token.is_stop,
        token.is_punct
    )
    

# Stopword Removal

filtered = [
    token.text
    for token in doc
    if not token.is_stop
]

print(filtered)


# Lemmatization - 

for token in doc:
    print(token.text, '->', token.lemma_)

# Eg - The → the
# cats → cat
# are → be
# running → run
# faster → fast


# POS Tagging - assign grammatical catgories 

for token in doc:
    print(token.text, token.pos_)

# Outputs - The DET
# dog NOUN
# runs VERB
# quickly ADV

# Common POS Tags - 
# POS - Meaning
# NOUN - Noun
# VERB - verb
# ADJ - Adjective
# ADV - adverb


# NER - Named Entity Recognition (NER)

doc = nlp(
    'Elon Musk founded Tesla in California'
)

for ent in doc.ents:
    print(ent.text, ent.label_)

# Elon Musk PERSON
# Tesla ORG
# California GPE

# Common Entity Labels 
# Label - Meaning
# PERSON, ORG - organisation, GPE - Country/City/State, DATE, MONEY - currency



# Sentence Segmentation - skip text into sentences

doc = nlp(
    'I love NLP. Transformers are amazing'
)

for sent in doc.sents:
    print(sent)


# Dependency Parsing - understand grammatical relationships

# Eg - Sentence - 'The dog chased the cat' - spaCy identifies - 'Subject, Object, Verb relationship'

doc = nlp('The dog chased the cat')

for token in doc:
    print(
        token.text,
        token.dep_,
        token.head.text
    )


# Word Similarity - supports embeddings

nlp = spacy.load('en_core_web_md')

doc1 = nlp('dog')
doc2 = nlp('cat')

print(doc1.similarity(doc2))

# Why Useful? - enables - semantic search, recommendation systems and clustering


# Custom NLP Pipelines -

print(nlp.pipe_names)

# Ouputs - ['tok2vec', 'tagger', 'parser', 'ner']


# Batch Processing - processing one text at a time in slow

texts = [
    'I love NLP',
    'Transformers are powerful',
    'spaCy is fast'
]

docs = nlp.pipe(texts)

for doc in docs:
    print(doc.text)


# Rule-Based Matching

from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)

pattern = [
    {'LOWER': 'machine'},
    {'LOWER': 'learning'}
]

matcher.add('ML_TERM', [pattern])

doc = nlp('I love machine learning')
matches = matcher(doc)
print(matches)

# real-life usage - resume parsing, entity extraction, keyword systems, compliance montioring


# Training Custom NER

# Eg - Train custom entities: 
#           Text - Entity
#           'Diabetes detected - DISEASE
#           'Aspirin prescribed - MEDICATION


