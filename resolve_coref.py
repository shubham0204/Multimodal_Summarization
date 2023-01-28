from datasets import load_dataset
from preprocessing import process_article
import torch
import spacy
import pickle
import neuralcoref

nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

dataset = load_dataset("cnn_dailymail", "3.0.0", split='test').with_format(type='torch')
loader = torch.utils.data.DataLoader(dataset)
print('Num samples: ', len(dataset))

def parse(doc):
  article = process_article( doc[ 'article' ][0] )
  target_summary = doc[ 'highlights' ][0]
  doc = nlp( article )
  return doc._.coref_resolved , target_summary

results = []
i = 0
for sample in loader:
    print('Processed', i + 1, 'sentences')
    results.append(parse(sample))
    i += 1
    if i % 1000 == 0:
      with open('coref_resolved/cnn_dailymail/{}_articles.pkl'.format(i), 'wb') as file:
        pickle.dump(results, file)
        print('Saved')
        results = []
