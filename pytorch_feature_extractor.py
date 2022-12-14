
from torch.utils.data import Dataset
from datasets import load_dataset
from features import sentence_length
from features import sentence_position
from features import sentence_num_proper_nouns
from features import sentence_num_numeric_terms
from sent_bert import get_sent_embedding
from similarity_metrics import cosine
from nltk import sent_tokenize
from preprocessing import process_article
import torch

class CNNFeatureExtractor( Dataset ):

    def __init__( self ):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        print( 'Active device: ' , device )
        self.dataset = load_dataset( "cnn_dailymail", "3.0.0", split='train').with_format(type='torch' )
        #self.dataset.to( device )

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        article = self.dataset[ item ]['article']
        target_summary = self.dataset[ item ]['highlights']
        sentences = sent_tokenize( process_article( article ) )
        f1 = sentence_length(sentences)
        f2 = sentence_position(sentences)
        f3 = sentence_num_proper_nouns(sentences)
        f4 = sentence_num_numeric_terms(sentences)
        scores = torch.mean(torch.tensor([f1, f2, f3, f4]), dim=0, keepdim=True)
        node_features = scores.transpose(1, 0)
        edge_features = torch.eye(n=num_sentences)
        for i in range(num_sentences):
            for j in range(i + 1):
                score = cosine(get_sent_embedding(sentences[i]), get_sent_embedding(sentences[j]))
                edge_features[i, j] = score
                edge_features[j, i] = score
        return node_features , edge_features , target_summary



