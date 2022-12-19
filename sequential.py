from datasets import load_dataset
from features import gusum_fused
from sent_bert import get_sent_embedding
from similarity_metrics import cosine
from preprocessing import process_article
from utils import get_summary
from metrics import compute_rouge_1
from nltk import sent_tokenize
import torch
import time
import numpy as np
import pickle

summaries = []
target_summaries = []

def parse( doc ):
    article = doc[ 'article' ]
    target_summary = doc[ 'highlights' ]
    sentences = sent_tokenize( process_article( article ) )
    num_sentences = len( sentences )
    
    scores = torch.mean( torch.tensor( gusum_fused( sentences ) ) , dim=0 , keepdim=True )
    node_features = scores.transpose( 1 , 0 )

    embeddings = []
    norms = []
    for i in range( num_sentences ):
        embeddings.append( get_sent_embedding( sentences[i] ) )
        norms.append( torch.norm( embeddings[i] , p=2 ) )
    embeddings = torch.stack( embeddings )
    norms = torch.tensor( norms )
    norms = torch.reshape( norms , [ norms.shape[0] , 1 ] )
    
    edge_features = embeddings @ embeddings.transpose( 1 , 0 )
    edge_features = edge_features / ( norms @ torch.transpose( norms , 1 , 0 ) )
    centrality = torch.reshape( node_features , ( num_sentences , ) ) * torch.sum( edge_features , dim=1 )
    summary = get_summary( sentences , centrality , k=3 )
    return summary , target_summary

if __name__ == '__main__':
    
    dataset = load_dataset( "cnn_dailymail" , "3.0.0" , split='test' ).with_format( type='torch' )
    print( 'Num samples: ' , len(dataset) )

    results = []
    samples = [ dataset[i] for i in range( len( dataset ) ) ]
    i = 10000
    num = 1000
    for sample in samples[i:]:
        print( 'Processed' , i + 1 , 'sentences' )
        results.append( parse( sample ) )
        i += 1

    with open( f'trial_02_summaries/10000_11490_summaries.pkl' , 'wb' ) as file:
        pickle.dump( results , file )


