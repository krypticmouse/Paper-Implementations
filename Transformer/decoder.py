import torch
from torch import nn

class Decoder_Block(nn.Module):
    def __init__(self, 
                d_model = 512, 
                num_heads = 8, 
                dropout = 0.1, 
                hidden_dim = 2048,
    ):
        super(Decoder_Block, self).__init__()

        self.mha = nn.MultiheadAttention(d_model, 
                                        num_heads = num_heads,
                                        dropout = dropout,)
        self.masked_mha = nn.MultiheadAttention(d_model, 
                                                num_heads = num_heads,
                                                dropout = dropout,)
        
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_query, enc_key, mask):
        res_x = x.clone()
        x = self.masked_mha(x, x, x, mask)
        res = self.norm_1(res_x + x)
        
        res_x = res
        x = self.mha(enc_query, enc_key, res)
        res = self.norm_2(res_x + x)

        res_x = res
        x = self.fc1(res)
        x = self.fc2(x)

        res = self.norm_3(res_x + x)
        return res

class Decoder(nn.Module):
    def __init__(self, 
                op_vocab_size,
                max_len = 128,
                embed_dim = 512,
                n_blocks = 6,
                d_model = 512
    ):
        super(Decoder, self).__init__()
        self.word_embeddings = nn.Embedding(op_vocab_size, embed_dim)
        self.postional_embeddings = nn.Embedding(max_len, embed_dim)

        self.decoder = nn.ModuleList([Decoder_Block() for _ in range(n_blocks)])

        self.output = nn.Linear(d_model, op_vocab_size)

    def forward(self, x, enc_op, mask):
        pos_embed = torch.arange(0, input[1]).expand(input[0], input[1])
        x = self.word_embeddings(input) + self.postional_embeddings(pos_embed)
        
        for l in self.decoder:
            x = l(x, enc_op, enc_op, mask)
        return torch.softmax(self.output(x))