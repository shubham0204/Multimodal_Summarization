from datasets import load_dataset
from features import gusum_fused
from sent_bert import get_sent_embedding
from preprocessing import process_article
from utils import get_summary
from nltk import sent_tokenize
import pandas as pd
import torch
import pickle

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

def parse( doc ):
    article = doc[0]
    target_summary = doc[1]

    sentences = sent_tokenize( article )
    num_sentences = len( sentences )
    
    scores = torch.mean( torch.tensor( gusum_fused( sentences ) ) , dim=0 , keepdim=True ).to( device )
    node_features = scores.transpose( 1 , 0 )

    embeddings = []
    norms = []
    for i in range( num_sentences ):
        embeddings.append( get_sent_embedding( sentences[i] ) )
        norms.append( torch.norm( embeddings[i] , p=2 ) )
    embeddings = torch.stack( embeddings ).to( device )
    norms = torch.tensor( norms ).to( device )
    norms = torch.reshape( norms , [ norms.shape[0] , 1 ] )
    
    edge_features = embeddings @ embeddings.transpose( 1 , 0 )
    edge_features = edge_features / ( norms @ torch.transpose( norms , 1 , 0 ) )
    centrality = torch.reshape( node_features , ( num_sentences , ) ) * torch.sum( edge_features , dim=1 )
    centrality = centrality.cpu()
    summary = get_summary( sentences , centrality , k=3 )
    return summary , target_summary

if __name__ == '__main__':
    
    ds = pd.read_csv( 'wikihow_data/cleaned_10000.csv' )
    headlines =  ds[ 'headline' ].values
    articles = ds[ 'text' ].values

    print( 'Num samples: ' , len(headlines) )

    results = []
    i = 0
    for sample in zip( articles , headlines ):
        print( 'Processed' , i + 1 , 'sentences' )
        results.append( parse( sample ) )
        i += 1
        if i % 1000 == 0:
          with open( 'summaries/trial_05B_summaries/{}_summaries.pkl'.format(i) , 'wb' ) as file:
            pickle.dump( results , file )
          print( 'Saved' )
          results = []
        
        

