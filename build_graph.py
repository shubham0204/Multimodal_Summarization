from datasets import load_dataset
from features import sentence_length
from features import sentence_position
from features import sentence_num_proper_nouns
from features import sentence_num_numeric_terms
from sent_bert import get_sent_embedding
from similarity_metrics import l2_norm
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
import torch_geometric

dataset = load_dataset( "cnn_dailymail" , "3.0.0" , split='test' )
article = dataset[ 1 ][ 'article' ]
sentences = article.split( "." )
sentences = [ sent for sent in sentences if len( sent.split() ) != 0 ]

# TODO: Add preprocessing code
f1 = sentence_length( sentences )
f2 = sentence_position( sentences )
f3 = sentence_num_proper_nouns( sentences )
f4 = sentence_num_numeric_terms( sentences )
scores = np.average( [ f1 , f2 , f3 , f4 ] , axis=0 )
node_features = np.expand_dims( scores , axis=1 )

num_sentences = len( sentences )
weights = np.zeros( shape=( num_sentences , num_sentences ) )
for i in range( num_sentences ):
    for j in range( num_sentences ):
        score = l2_norm( get_sent_embedding( sentences[i] ) , get_sent_embedding( sentences[j] ) )
        weights[ i , j ] = score

xx , yy = np.meshgrid( np.arange( num_sentences ) , np.arange( num_sentences ) )
xx = np.reshape( xx , newshape=( num_sentences**2 , ) )
yy = np.reshape( yy , newshape=( num_sentences**2 , ) )
edge_index = np.array( [ xx , yy ] )

node_features = torch.tensor( node_features , dtype=torch.float32 )
edge_features = torch.tensor( weights , dtype=torch.float32 )
edge_index = torch.tensor( edge_index , dtype=torch.float32 )

data = torch_geometric.data.Data(
    x=node_features,
    edge_index=edge_index ,
    edge_attr=edge_features ,
    is_directed=False
)

graph = torch_geometric.utils.to_networkx( data , to_undirected=True )

pos = nx.spring_layout( graph , seed=1045 )
nx.draw_networkx_nodes( graph , pos , node_color="indigo" )
nx.draw_networkx_edges(
    graph ,
    pos ,
    edge_cmap=plt.cm.plasma ,
    edge_color=np.reshape( weights , ( 18**2 , ) )
)
plt.show()




