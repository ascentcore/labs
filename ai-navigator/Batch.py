import torch
import numpy as np
from torch.autograd import Variable

def nopeak_mask(size):

    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)

    return np_mask.tolist()[0]

def create_masks(src, trg):

    batch_size = src.size(dim=0)

    src_dim_1 = src.size(dim=1)
    src_mask_elements = [True for _ in range(src_dim_1)]
    src_mask = [[src_mask_elements] for _ in range(batch_size)]

    trg_dim_1 = trg.size(dim=1)
    trg_mask_elements = nopeak_mask(trg_dim_1)
    trg_mask = [trg_mask_elements for _ in range(batch_size)]

    return src_mask, trg_mask
