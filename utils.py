import numpy as np
import torch


def get_summary( sentences , ranks , k=3 ):
    sentences = np.array( sentences )
    return sentences[ np.argsort( ranks ) ][ -1 : -(k+1) : -1 ]