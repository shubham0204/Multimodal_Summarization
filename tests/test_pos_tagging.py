import nltk

nltk.download( 'averaged_perceptron_tagger' )

sent = "Ashwin is a hardworking boy of 20 years."
pos = nltk.pos_tag( sent.split() )
print( pos )