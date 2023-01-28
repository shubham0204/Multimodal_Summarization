from metrics import evaluate
import numpy as np
import os
import pprint
import pickle

summaries_dir = 'summaries/trial_05B_summaries'
names = os.listdir( summaries_dir )
summaries = []
for name in names:
    file = open( os.path.join( summaries_dir , name ) , 'rb' )
    result = pickle.load( file )
    for i in range( len( result ) ):
        summaries.append( result[i] )
    file.close()
summaries = np.array( summaries )

hypothesis = summaries[ : , 0 ]
references = summaries[ : , 1 ]

print( 'Processing completed.' )
pprint.pprint( evaluate(references, hypothesis) )