import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("AI is transforming the world.")
for token in doc:
    print(token.text, token.pos_)
