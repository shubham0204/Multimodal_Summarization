import torch

from similarity_metrics import cosine

vec1 = torch.tensor( [ 1.0 , 2.0 ] )
vec2 = torch.tensor( [ 2.0 , 1.0 ] )
print( cosine( vec1 , vec2 ) )

vec1 = torch.tensor( [ 1.0 , 2.0 ] )
vec2 = torch.tensor( [ 1.0 , 2.0 ] )
print( cosine( vec1 , vec2 ) )

vec1 = torch.tensor( [ 1.0 , 1.0 ] )
vec2 = torch.tensor( [ -1.0 , 1.0 ] )
print( cosine( vec1 , vec2 ) )