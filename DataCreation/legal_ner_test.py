# This will work inside your new, activated environment

from flair.data import Sentence
from flair.models import SequenceTagger

print("Flair imported successfully!")

# Load the pre-trained English NER model
tagger = SequenceTagger.load("ner-english-large")

# Your text
text = "Section 319 Cr.P.C. contemplates a situation where the evidence adduced by the prosecution for Respondent No.3-G. Sambiah on 20th June 1984"

# Create a Flair Sentence object
sentence = Sentence(text)

# Predict the entities
tagger.predict(sentence)

# Print the results
print("\nFound the following entities:")
for entity in sentence.get_spans('ner'):
    print(f"- '{entity.text}' [{entity.tag}]")