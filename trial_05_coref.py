from datasets import load_dataset
from features import gusum_fused
from sent_bert import get_sent_embedding
from preprocessing import process_article
from utils import get_summary
from nltk import sent_tokenize
from allennlp.predictors.predictor import Predictor
import torch
import pickle

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
predictor = Predictor.from_path(model_url)

def parse( doc ):
    article = doc[ 'article' ][0]
    print( type( article) )
    article = predictor.coref_resolved( str(article) )
    target_summary = doc[ 'highlights' ][0]
    print( type( target_summary ) )
    sentences = sent_tokenize( process_article( article ) )
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
    
    dataset = load_dataset( "cnn_dailymail" , "3.0.0" , split='test' ).with_format( type='torch' )
    loader = torch.utils.data.DataLoader( dataset )
    print( 'Num samples: ' , len(dataset) )

    results = []
    i = 0
    for sample in loader:
        print( 'Processed' , i + 1 , 'sentences' )
        results.append( parse( sample ) )
        i += 1
        if i % 1000 == 0:
          with open( 'Multimodal_Summarization/summaries/trial_05_summaries/{}_summaries.pkl'.format(i) , 'wb' ) as file:
            pickle.dump( results , file )
          print( 'Saved' )
          results = []
        
        


