from nltk import pos_tag

def sentence_length( sentences ):
    num_words = []
    for sent in sentences:
        num_words.append( len( sent.split() ) )
    max_num_words = max( num_words )
    relative_lengths = [ ( ni / max_num_words ) for ni in num_words ]
    return relative_lengths

def sentence_position( sentences ):
    positions = []
    N = len( sentences )
    for i in range( len( sentences ) ):
        if i == 0 or i == N - 1:
            positions.append( 1.0 )
        else:
            positions.append( ( N - i ) / N )
    return positions

def sentence_num_proper_nouns( sentences ):
    return __sentence_num_pos_tags(sentences, tag='NNP')

def sentence_num_numeric_terms( sentences ):
    return __sentence_num_pos_tags(sentences, tag='CD')

def __sentence_num_pos_tags(sentences, tag):
    rate_pos_terms = []
    for sent in sentences:
        words = sent.split()
        pos = pos_tag(words)
        num_terms = sum([1 for x in pos if x[1] == tag])
        rate_pos_terms.append( num_terms / len( words ))
    return rate_pos_terms