import spacy
import re
import pandas as pd
from nltk.corpus import stopwords, wordnet
from unifyingSynm import synonym_replacement
from NegationHandeling import negation_replacement


#passing the "en_core_web_sm" model into nlp pipline. 
nlp = spacy.load("en_core_web_sm")

#df = pd.read_csv('Data/Sample-Training-Manually-Classified.csv')
#df.drop(['id'], axis=1, inplace=True)

# customizing stopwords list to avoid removing them in the process of stopwords removal. 
# list in non_stopwords will be used for negation handling
stopwords = stopwords.words("english")
non_stopwords = ['not','no','never','none','nor','hadn','mustn',"didn't",'doesn',"hadn't","mustn't",
                 'mightn','haven',"aren't","haven't",'weren','didn',"couldn't","doesn't","hasn't",'isn',
                 'wasn','needn','mustn',"weren't",'don','couldn','wouldn',"mightn't","wouldn't","don't",
                 'ain',"shouldn't",'aren',"isn't","needn't","wasn't",'shouldn','hasn',"won't"]
my_stopwords = set([word for word in stopwords if word not in non_stopwords])
#print("Total number of stopwords is:" + " " + str(len(my_stopwords)))
#print(my_stopwords)

def preprocessing(text):

    # Unifying Synonym
    #text = synonym_replacement(text)

    # convert to lowercase
    text = text.lower()

    # remove punctuation, numbers
    text = ''.join(word for word in re.sub("[^a-zA-Z]", " ", text))


    # remove stopwords
    text = ' '.join([word for word in text.split() if word not in my_stopwords])

    # Negation handeling
    text = negation_replacement(text)

    return ''.join([word for word in text])


# df['processed'] = df['body'].apply(preprocessing)
# df.to_csv('processed.csv')