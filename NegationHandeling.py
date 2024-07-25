import pandas as pd, numpy as np
import spacy
import nltk
from nltk.corpus import wordnet, stopwords

#passing the "en_core_web_sm" model into nlp pipline. 
nlp = spacy.load("en_core_web_sm")

# df = pd.read_csv('Data/Sample-Training-Manually-Classified.csv')
# df.drop(['id'], axis=1, inplace=True)
#print(df.head())

# customizing stopwords list to avoid removing them in the process of stopwords removal. 
# list in non_stopwords will be used for negation handling
#stopwords = stopwords.words("english")
non_stopwords = ['not','no','never','none','nor','hadn','mustn',"didn't",'doesn',"hadn't","mustn't",
                 'mightn','haven',"aren't","haven't",'weren','didn',"couldn't","doesn't","hasn't",'isn',
                 'wasn','needn','mustn',"weren't",'don','couldn','wouldn',"mightn't","wouldn't","don't",
                 'ain',"shouldn't",'aren',"isn't","needn't","wasn't",'shouldn','hasn',"won't"]
#my_stopwords = set([word for word in stopwords if word not in non_stopwords])


#in negation handling, we'r removing the negative words, like not, no, etc.,
#and substituting adj/verbs after the negative words with it's antonym
def negation_handling(unigram):    
    antonyms = []

    #check if the word after the "not" is adj or verb
    #we use the partof speech to tage each workd and itentify it's type. 
    if nlp(unigram)[0].pos_ in ['ADJ', 'VERB']:

        #then find the antonym of this word and add to the list
        for syn in wordnet.synsets(unigram):
            for l in syn.lemmas():
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())
    
    # keep unique antonyms
    antonyms = set(antonyms)

    # return first adjective antonym
    # to ensure that the word is being substituted w/ its antonym
    for antonym in antonyms:
        # Since we want to avoid substituting for example a verb with an
        # adjective, the POS tagging must be the same
        if nlp(antonym)[0].pos_ == nlp(unigram)[0].pos_:
            return antonym
        
    # if no adj-verb antonym found, return original word
    return unigram


def negation_replacement(sentence):
    # negation handling
    #check bigrams, if firt word is one of the negative words,
    #take the word after it and pass it to the negation_handling function. 
    
    file = nlp(sentence)
    sentence = [token.lemma_ for token in file]


    antonyms = {}
    for idx, (first, second) in enumerate(zip(sentence, sentence[1:])):
        if first in non_stopwords:
            antonyms[idx - len(antonyms.keys())] = negation_handling(second)

    for key in antonyms.keys():
        sentence[key] = antonyms[key]
        sentence.pop(key + 1)

    return ' '.join([word for word in sentence])

# df['negated'] = df['body'].apply(negation_replacement)
# #print(df.head)
# df.to_csv('negated.csv')