import numpy as np

def get_summary( sentences , ranks , k=3 ):
    sentences = np.array( sentences )
    top_k_sents = sentences[ np.argsort( ranks ) ][ -1 : -(k+1) : -1 ]
    if type( top_k_sents ) == np.ndarray:
      return '.'.join( top_k_sents.tolist() )
    else:
      return top_k_sents