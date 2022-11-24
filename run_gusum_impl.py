from datasets import load_dataset
from features import sentence_length
from features import sentence_position
from features import sentence_num_proper_nouns
from features import sentence_num_numeric_terms
from sent_bert import get_sent_embedding
from similarity_metrics import l2_norm
from utils import get_summary
import numpy as np
import torch
import time

dataset = load_dataset( "cnn_dailymail" , "3.0.0" , split='test' )

t1 = time.time()
article = dataset[ 1 ][ 'article' ]
sentences = article.split( "." )
sentences = [ sent for sent in sentences if len( sent.split() ) != 0 ]

# TODO: Add preprocessing code
# Add feature: sentence length deviation
f1 = sentence_length( sentences )
f2 = sentence_position( sentences )
f3 = sentence_num_proper_nouns( sentences )
f4 = sentence_num_numeric_terms( sentences )
scores = np.average( [ f1 , f2 , f3 , f4 ] , axis=0 )
node_features = np.expand_dims( scores , axis=1 )

num_sentences = len( sentences )
weights = np.identity( n=num_sentences )
for i in range( num_sentences ):
    for j in range( i + 1 ):
        score = l2_norm( get_sent_embedding( sentences[i] ) , get_sent_embedding( sentences[j] ) )
        weights[ i , j ] = score
        weights[ j , i ] = score

node_features = torch.tensor( node_features , dtype=torch.float32 )
edge_features = torch.tensor( weights , dtype=torch.float32 )

centrality = node_features.resize( 18 , ) * torch.sum( edge_features , dim=1)
centrality = centrality.cpu().detach().numpy()
summary = get_summary( sentences , centrality , k=3 )
print( summary )

print( 'Time taken {}'.format( time.time() - t1 ) )

