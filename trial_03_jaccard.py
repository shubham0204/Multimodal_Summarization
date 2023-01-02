from datasets import load_dataset
from features import gusum_fused
from sent_bert import get_sent_embedding
from similarity_metrics import jaccard_distance
from preprocessing import process_article
from utils import get_summary
from nltk import sent_tokenize
import torch
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
    for i in range( num_sentences ):
        embeddings.append( get_sent_embedding( sentences[i] ) )
    
    edge_features = torch.eye( num_sentences )
    for i in range( num_sentences ):
        for j in range( i , num_sentences ):
            edge_features[ i , j ] = jaccard_distance( embeddings[ i ] , embeddings[ j ] )
            edge_features[ j , i ] = edge_features[ i , j ]
            
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
    for sample in samples[i:i+num]:
        print( 'Processed' , i + 1 , 'sentences' )
        results.append( parse( sample ) )
        i += 1

    with open( f'summaries/trial_03_summaries/10000_11490.pkl' , 'wb' ) as file:
        pickle.dump( results , file )


