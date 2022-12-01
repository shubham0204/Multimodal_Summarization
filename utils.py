import numpy as np

def get_summary( sentences , ranks , k=3 ):
    sentences = np.array( sentences )
    top_k_sents = sentences[ np.argsort( ranks ) ][ -1 : -(k+1) : -1 ].tolist()
    return '.'.join( top_k_sents )