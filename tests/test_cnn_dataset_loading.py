from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset( "cnn_dailymail" , "3.0.0" , split='test' )
print( dataset[0] )