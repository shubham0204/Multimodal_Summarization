from datasets import load_dataset

dataset = load_dataset( "cnn_dailymail" , "3.0.0" , split='test' )
df = dataset.to_pandas()
print( df )