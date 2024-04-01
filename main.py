#pip install -U spacy -q
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import json
from collections import Counter

nlp = spacy.blank("en")
#nlp = spacy.blank("en_core_sm") # load a new spacy model
db = DocBin() # create a DocBin object

f = open('training_data.json')
TRAIN_DATA = json.load(f)

for text, annot in tqdm(TRAIN_DATA['annotations']):
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in annot["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents
    db.add(doc)

db.to_disk("./training_data.spacy") # save the docbin object


# ! python -m spacy init config config.cfg --lang en --pipeline ner --optimize efficiency


# ! python -m spacy train config.cfg --output ./ --paths.train ./training_data.spacy --paths.dev ./training_data.spacy

nlp_ner = spacy.load("/content/model-best")


doc = nlp_ner("TESTING DATA")


entity_counts = Counter([ent.label_ for ent in doc2.ents])

total_entities = sum(entity_counts.values())

normalized_scores = {label: round(count / total_entities, 3) for label, count in entity_counts.items()}