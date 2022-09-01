import torch
import json
import numpy as np

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset_files, w_s, t_s, jump):
        self.dataset = []
        # self.start_embed = np.zeros(4, dtype=float)

        for dataset_file in dataset_files:
            data = json.loads(open(dataset_file).read())
            for i in range(len(data)):
                if (len(data) - i >= w_s+jump+1):
                    current_position = data[i]
                    source = [np.subtract(data[i+acc+1], current_position) for acc in range(w_s)]
                    source.append(np.subtract(data[i+w_s+jump], current_position))
                    target = [np.subtract(data[i+acc+w_s+1], current_position) for acc in range(t_s)]
                    # target.insert(0, self.start_embed)

                    self.dataset.append((source, target))

    def __getdataset__(self):
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        in_data, out_data = self.dataset[idx]

        input = torch.tensor(in_data, dtype=torch.float32)
        output = torch.tensor(out_data, dtype=torch.float32)

        return input, output
