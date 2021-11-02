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

    def forward(self, x, enc_query, enc_key):
        res_x = x.clone()
        x = self.masked_mha(x, x, x)
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
    def __init__(self, n_blocks = 6):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList([Decoder_Block() for _ in range(n_blocks)])

    def forward(self, x, enc_op):
        for l in self.decoder:
            x = l(x, enc_op, enc_op)
        return x