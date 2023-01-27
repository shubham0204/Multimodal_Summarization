import pandas as pd

ds = pd.read_csv( 'wikihow_data/wikihowAll.csv' )
ds.dropna( inplace=True )
ds = ds.sample( 10000 )

ds.drop( [ 'title' ] , axis=1 , inplace=True )

def func( x ):
    from preprocessing import process_article
    return process_article( x )

res = ds[ 'text' ].apply( func )
ds[ 'text' ] = res

ds.to_csv( 'wikihow_data/cleaned_10000.csv' )