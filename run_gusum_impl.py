from datasets import load_dataset
from features import sentence_length
from features import sentence_position
from features import sentence_num_proper_nouns
from features import sentence_num_numeric_terms
from features import gusum_fused
from sent_bert import get_sent_embedding
from similarity_metrics import cosine
from preprocessing import process_article
from utils import get_summary
from metrics import compute_rouge_1
from nltk import sent_tokenize
from torch.utils.data import DataLoader
import torch
import time
import numpy as np
import pickle
import torch.multiprocessing as mp
import pprint

summaries = []
target_summaries = []

def parse( doc ):
    print( 'Method called' )
    t1 = time.time() ; 
    article = doc[ 'article' ]
    target_summary = doc[ 'highlights' ]
    sentences = sent_tokenize( process_article( article ) )
    num_sentences = len( sentences )
    
    scores = torch.mean( torch.tensor( gusum_fused( sentences ) ) , dim=0 , keepdim=True )
    node_features = scores.transpose( 1 , 0 )

    embeddings = []
    for sentence in sentences:
        embeddings.append( get_sent_embedding( sentence ) )
    embeddings = torch.stack( embeddings )
    
    edge_features = embeddings @ embeddings.transpose( 1 , 0 )
    centrality = torch.reshape( node_features , ( num_sentences , )) * torch.sum( edge_features , dim=1 )
    centrality = centrality.cpu().detach().numpy()
    summary = get_summary( sentences , centrality , k=3 )
    return ( summary , target_summary )
    
if __name__ == '__main__':
    global dataset
    dataset = load_dataset( "cnn_dailymail" , "3.0.0" , split='test' ).with_format( type='torch' )
    # print( 'Num samples: ' , len(dataset) )
    import time
    t1 = time.time()
    indices = [ dataset[i] for i in range( 100 ) ]
    pool = mp.get_context( 'spawn' ).Pool( mp.cpu_count() - 1 )
    results = pool.map_async( parse , indices ).get()
    pool.close()
    pool.join()
    print( results[0][0] )
    print( '--------' )
    print( results[0][1] )
    print( time.time() - t1 )

"""
with open( '500_1000_summaries.pkl' , 'wb' ) as file:
    pickle.dump( summaries , file )
score = compute_rouge_1( [ summary ] , [ target_summary ] )
"""


