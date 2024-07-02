import pandas as pd
import pickle

df = pd.read_csv('Data/Test-Validation.csv')
df.drop(['ID'], axis=1, inplace=True)
#print(df.head())

df['classified'] = None

texts = df['body']

load_pickle = open('RandomForest.pickle', 'rb')
rfc = pickle.load(load_pickle)

def classifier():
    list = []
    for text in texts:
        text1 = [str(text)]
        pred = rfc.predict(text1)
        list.append(pred[0])
    return list

prediction = classifier()
df['classified'] = prediction

df.to_csv('result.csv')