# spaCy - used for - NER, POS Tagging, Preprocessing. (Traditional NLP library)
 
import spacy 

nlp = spacy.load('en_core_web_sm')

doc = nlp('Apple hired John')

for ent in doc.ents:
    print(ent.text, ent.label_)
