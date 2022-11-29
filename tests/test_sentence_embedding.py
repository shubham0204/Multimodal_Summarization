from sent_bert import get_sent_embedding
from similarity_metrics import cosine

vec1 = get_sent_embedding( "Abdul Kalam was an Indian scientist and engineer who worked primarily at ISRO and DRDO." )
vec2 = get_sent_embedding( "He had a degree in aeronautical engineering from the Madras institute of technology." )
print( cosine( vec1 , vec2 ) ) 

vec3 = get_sent_embedding( "Abdul Kalam was an Indian scientist and engineer who worked primarily at ISRO and DRDO." )
vec4 = get_sent_embedding( "Abdul Kalam had a degree in aeronautical engineering from the Madras institute of technology." )
print( cosine( vec3 , vec4 ) )