import time
import torch.multiprocessing

from torch.utils.data import DataLoader
from pytorch_feature_extractor import CNNFeatureExtractor

def main():
    loader = DataLoader( dataset=CNNFeatureExtractor() , num_workers=0  )
    t1 = time.time()
    num = 0
    for sample in loader:
        print( sample )
        print('Time taken : ', time.time() - t1)
        # break
    print( 'Time taken : ' , time.time() - t1 )

if __name__ == '__main__':
    torch.multiprocessing.set_start_method( 'spawn' )
    main()