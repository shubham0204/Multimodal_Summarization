from datasets import load_dataset
from features import gusum_fused
from sent_bert import get_sent_embedding
from preprocessing import process_article
from utils import get_summary
from nltk import sent_tokenize
import torch
import pickle
import yake

kw_extractor = yake.KeywordExtractor()

def parse( doc ):
    article = process_article( doc[ 'article' ] )
    target_summary = doc[ 'highlights' ]
    
    keywords = kw_extractor.extract_keywords( article )
    keywords = [ kw[0] for kw in keywords ]
    
    sentences = sent_tokenize( article )
    num_sentences = len( sentences )
    
    embeddings = []
    kw_weights = []
    norms = []
    for i in range( num_sentences ):
        kw_weight = sum( [ 1 if kw in sentences[i] else 0 for kw in keywords ] ) / len( keywords )
        embeddings.append( get_sent_embedding( sentences[i] ) )
        norms.append( torch.norm( embeddings[i] , p=2 ) )
        kw_weights.append( [kw_weight] )
    embeddings = torch.stack( embeddings )
    norms = torch.tensor( norms )
    norms = torch.reshape( norms , [ norms.shape[0] , 1 ] )
    
    kw_weights = torch.tensor( kw_weights )
    features = torch.tensor( gusum_fused( sentences ) ).transpose( 1 , 0 )
    features = torch.cat( [ kw_weights , features ] , dim=1 ) 
    features = features.transpose( 1 , 0 )
    scores = torch.mean( features , dim=0 , keepdim=True )
    node_features = scores.transpose( 1 , 0 )
    
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
    i = 6000
    num = 7000
    for sample in samples[i:i+num]:
        print( 'Processed' , i + 1 , 'sentences' )
        results.append( parse( sample ) )
        i += 1

    with open( f'summaries/trial_04_summaries/6000_7000.pkl' , 'wb' ) as file:
        pickle.dump( results , file )


