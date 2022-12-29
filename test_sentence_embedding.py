from sent_bert import get_sent_embedding
from similarity_metrics import cosine , jaccard_distance , l2_norm

vec1 = get_sent_embedding( "Abdul Kalam was an Indian scientist and engineer who worked primarily at ISRO and DRDO." )
vec2 = get_sent_embedding( "He had a degree in aeronautical engineering from the Madras institute of technology." )
print( 1 - jaccard_distance( vec1 , vec2 ) ) 
print( cosine( vec1 , vec2 ) )
print( l2_norm( vec1 , vec2 ) )

vec3 = get_sent_embedding( "Abdul Kalam was an Indian scientist and engineer who worked primarily at ISRO and DRDO." )
vec4 = get_sent_embedding( "Abdul Kalam had a degree in aeronautical engineering from the Madras institute of technology." )
print( 1 - jaccard_distance( vec3 , vec4 ) )
print( cosine( vec3 , vec4 ) )
print( l2_norm( vec3 , vec4 ) )