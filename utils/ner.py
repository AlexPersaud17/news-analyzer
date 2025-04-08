import spacy

nlp = spacy.load("en_core_web_sm")

def ner_get_entities(data):
    entities = []
    spacy_doc = nlp(data["article"])
    for ent in spacy_doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })
    return entities