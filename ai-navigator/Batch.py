import torch

def create_masks(src, trg, opt):

    src_tensor = torch.Tensor(src)
    batch_size = src_tensor.size(dim=0)

    src_dim_1 = src_tensor.size(dim=1)
    src_mask_elements = [True for _ in range(src_dim_1)]
    src_mask = [[src_mask_elements] for _ in range(batch_size)]

    trg_mask_elements = [[True, False], [True, True]]
    trg_mask = [trg_mask_elements for _ in range(batch_size)]
    
    return src_mask, trg_mask
