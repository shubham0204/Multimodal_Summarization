from nltk import sent_tokenize
from features import gusum_fused
import pandas as pd
import pickle


ds = pd.read_csv('wikihow_data/cleaned_10000.csv')
headlines = ds['headline'].values
articles = ds['text'].values

print('Num samples: ', len(headlines))

def parse( article ):
    sentences = sent_tokenize(article)
    return gusum_fused(sentences)

results = []
i = 0
for sample in articles:
    print('Processed', i + 1, 'sentences')
    results.append(parse(sample))
    i += 1
    
with open( "features_wikihow_gusum.pkl" , "wb" ) as file:
    pickle.dump( results , file )
