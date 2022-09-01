import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from Models import get_model
from Batch import create_masks
from Loader import CustomDataset
from autoencoder.AE import AE


def train_model(model, opt):

    print("training model...")
    model.train()

    for epoch in range(opt.epochs):
        print('--------------EPOCH ', epoch)
        total_loss = 0
        for batch_idx, (train_features, train_labels) in enumerate(opt.train):

            train_features = [feat.tolist() for feat in train_features]
            train_labels = [label.tolist() for label in train_labels]

            train_features_2 = np.transpose(np.array(train_features), (1, 0, 2))
            train_labels_2 = np.transpose(np.array(train_labels), (1, 0, 2))

            src = torch.Tensor(train_features_2)
            trg = torch.Tensor(train_labels_2)
            
            b_s = src.size(dim=0)
            src_reshape = torch.reshape(src, (b_s, (opt.w_s + 1)  * 4))
            # src_mask, trg_mask = create_masks(src, trg)

            opt.optimizer.zero_grad()
            # preds = model(src, trg, src_mask, trg_mask)

            # print('src_reshape', src_reshape)
            preds = model(src_reshape)
            preds_reshape = torch.reshape(preds, (b_s, opt.t_s, 4))
            # print('preds_reshape', preds_reshape)

            loss_function = nn.MSELoss()
            loss = loss_function(preds_reshape, trg)
            if batch_idx % 10 == 0:
                print('loss', loss.item())
                torch.save(model.state_dict(), f'./saved_model/model.pt')
            total_loss += loss.item()

            loss.backward()
            opt.optimizer.step()


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-epochs', type=int, default=1000)
    parser.add_argument('-d_model', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=3)
    parser.add_argument('-heads', type=int, default=2)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=32)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-w_s', type=int, default=10)
    parser.add_argument('-t_s', type=int, default=11)
    parser.add_argument('-jump', type=int, default=20)
    opt = parser.parse_args()
    opt.device = -1
    if opt.device == 0:
        assert torch.cuda.is_available()

    return opt


def main():

    opt = get_opts()

    # dataset = CustomDataset(['./dataset/train/run1.json', './dataset/train/run2.json',
    #                         './dataset/train/run3.json', './dataset/train/run4.json',
    #                         './dataset/train/run5.json', './dataset/train/run6.json',
    #                         './dataset/train/run7.json', './dataset/train/run8.json'], opt.w_s, opt.t_s, opt.jump)
    dataset = CustomDataset(['./dataset/train/run1.json'], opt.w_s, opt.t_s, opt.jump)

    training_data = dataset.__getdataset__()
    len_training_data = dataset.__len__()
    print('training_data', training_data[0:2])
    print('len', len_training_data)
    train_dataloader = DataLoader(
        training_data, batch_size=opt.batchsize, shuffle=True)
    opt.train = train_dataloader

    # model = get_model(opt)
    # opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    model = AE()
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    train_model(model, opt)


if __name__ == "__main__":
    main()
