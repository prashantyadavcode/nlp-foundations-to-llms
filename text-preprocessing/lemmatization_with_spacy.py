# Lemmatization with spaCy

import spacy

nlp = spacy.load('en_core_web_sm')
text = 'The cats are running faster'

doc = nlp(text)
lemmas = [token.lemma_ for token in doc]
print(lemmas)