import numpy as np
import torch

def l1_norm( vec1 , vec2 ):
    return torch.norm( vec1 - vec2 , p=1 )

def l2_norm( vec1 , vec2 ):
    return torch.norm( vec1 - vec2 , p=2 )

def jaccard_distance( vec1 , vec2 ):
    max_sum = torch.sum( torch.maximum( vec1 , vec2 ) )
    min_sum = torch.sum( torch.minimum( vec1 , vec2 ) )
    return 1 - (min_sum / max_sum)

def cosine( vec1 , vec2 ):
    return torch.dot( vec1 , vec2 ) / ( torch.norm( vec1 , p=2 ) * torch.norm( vec2 , p=2 ) )

