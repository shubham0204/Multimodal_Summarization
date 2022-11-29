from datasets import load_dataset
from features import sentence_length
from features import sentence_position
from features import sentence_num_proper_nouns
from features import sentence_num_numeric_terms
from sent_bert import get_sent_embedding
from similarity_metrics import cosine
from utils import get_summary
from metrics import compute_rouge_1
from torch.utils.data import DataLoader
import numpy as np
import torch
import time

dataset = load_dataset( "cnn_dailymail" , "3.0.0" , split='test' ).with_format( type='torch' )

t1 = time.time()
article = dataset[ 1 ][ 'article' ]
target_summary = dataset[ 1 ][ 'highlights' ]
sentences = article.split( "." )
sentences = [ sent for sent in sentences if len( sent.split() ) != 0 ]
num_sentences = len( sentences )

# TODO: Add preprocessing code
# Add feature: sentence length deviation
f1 = sentence_length( sentences )
f2 = sentence_position( sentences )
f3 = sentence_num_proper_nouns( sentences )
f4 = sentence_num_numeric_terms( sentences )

scores = torch.mean( torch.tensor( [ f1 , f2 , f3 , f4 ] ) , dim=0 , keepdim=True )
node_features = scores.transpose( 1 , 0 )

edge_features = torch.eye( n=num_sentences )
for i in range( num_sentences ):
    for j in range( i + 1 ):
        score = cosine( get_sent_embedding( sentences[i] ) , get_sent_embedding( sentences[j] ) )
        edge_features[ i , j ] = score
        edge_features[ j , i ] = score

centrality = node_features.resize( 18 , ) * torch.sum( edge_features , dim=1)
centrality = centrality.cpu().detach().numpy()

summary = get_summary( sentences , centrality , k=3 )
score = compute_rouge_1( [ summary ] , [ target_summary ] )

print( 'Time taken {} and score {}'.format( time.time() - t1 , score ) )

