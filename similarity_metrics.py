import numpy as np

def l1_norm( vec1 , vec2 ):
    return np.linalg.norm( vec1 - vec2 , ord=1 )

def l2_norm( vec1 , vec2 ):
    return np.linalg.norm( vec1 - vec2 , ord=2 )

def jaccard_score( vec1 , vec2 ):
    max_sum = np.sum( np.maximum( vec1 , vec2 ) )
    min_sum = np.sum( np.minimum( vec1 , vec2 ) )
    return min_sum / max_sum

