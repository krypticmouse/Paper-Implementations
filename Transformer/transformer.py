import torch
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, 
                inp_vocab_size, 
                op_vocab_size,
                max_len = 128,
                embed_dim = 512,
                n_blocks = 6,
        ):
        self.encoder = Encoder(inp_vocab_size,
                               max_len = max_len,
                               embed_dim = embed_dim,
                               n_blocks = n_blocks)
        
        self.decoder = Decoder(op_vocab_size,
                               max_len = max_len,
                               embed_dim = embed_dim,
                               n_blocks = n_blocks)

    def forward(self, input, output):
        N, op_len = output.shape
        output_mask = torch.tril(torch.ones((op_len, op_len))).expand(N, 1, op_len, op_len)

        enc_op = self.encoder(input)
        return self.decoder(output, enc_op, output_mask)
