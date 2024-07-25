import pandas as pd, numpy as np
import re, string
import spacy
from nltk.corpus import stopwords, wordnet

#passing the "en_core_web_sm" model into nlp pipline. 
nlp = spacy.load("en_core_web_sm")


# df = pd.read_csv('Data/Sample-Training-Manually-Classified.csv')
# df.drop(['id'], axis=1, inplace=True)
#print(df.head())


# Getting the synonyms of the selected words
def get_synonyms(word):
    synonyms = set()

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            if synonym != '':
                synonyms.add(synonym)

    if word in synonyms:
        synonyms.remove(word)

    return list(synonyms)


# Creating new sentences by replacing the synonyms of nouns, verbs and adjectives
def synonym_replacement(sentence):
    sentence = sentence.split()
    returned_sentence = []

    for token in sentence:

        # If the word is a noun, verb or adjective
        if nlp(token)[0].pos_ in ['NOUN', 'VERB', 'ADJ']:

            synonyms = get_synonyms(token)

            synonym_found = 0
            for synonym in synonyms:
                # Since many different synonyms are returned, only those that have the same POS tag are substituted,
                # so that a noun for instance, is not substituted by a verb
                if (nlp(synonym)[0].pos_ == nlp(token)[0].pos_) & (len(synonym) >= 3):
                    returned_sentence.append(synonym)
                    synonym_found = 1
                    break

            if synonym_found == 0:
                returned_sentence.append(token)
        else:
            returned_sentence.append(token)

    return ' '.join([word for word in returned_sentence])


# df['unified'] = df['body'].apply(synonym_replacement)
# #print(df.head)
# df.to_csv('unified.csv')
