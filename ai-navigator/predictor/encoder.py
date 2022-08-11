from torch import nn

from layers import EncoderLayer, get_clones
from normalisation import Norm

class Encoder(nn.Module):
    # def __init__(self, vocab_size, d_model, N, heads):
    def __init__(self, d_model, N, heads):
        super().__init__()
        
        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        # x = self.embed(src)
        # x = self.pe(x)
        x = src
        for i in range(self.N):
            x = self.layers[i](x, mask)
        
        return self.norm(x)
