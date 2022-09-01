import torch

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
         
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(44, 36),
            torch.nn.Linear(36, 24),
            torch.nn.Linear(24, 18),
        )
         
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(18, 24),
            torch.nn.Linear(24, 36),
            torch.nn.Linear(36, 44),
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
