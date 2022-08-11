from torch import nn

from layers import DecoderLayer, get_clones
from normalisation import Norm

class Decoder(nn.Module):
    # def __init__(self, vocab_size, d_model, N, heads):
    def __init__(self, d_model, N, heads):
        super().__init__()

        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        # self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        # x = self.embed(trg)
        # x = self.pe(x)
        x = trg
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
    
        return self.norm(x)
