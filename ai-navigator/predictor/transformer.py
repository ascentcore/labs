from torch import nn

from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, d_model, N, heads):
    # def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()

        # self.encoder = Encoder(src_vocab, d_model, N, heads)
        # self.decoder = Decoder(trg_vocab, d_model, N, heads)
        # self.out = nn.Linear(d_model, trg_vocab)
        self.encoder = Encoder(d_model, N, heads)
        self.decoder = Decoder(d_model, N, heads)
        self.out = nn.Linear(d_model, 4)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)

        return output
# we don't perform softmax on the output as this will be handled 
# automatically by our loss function
