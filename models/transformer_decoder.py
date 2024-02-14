import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerDecoder

from 


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 d_feedforward=2048,
                 n_head=8,
                 n_layers=6,
                 dropout=0.5):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_feedforward, dropout, batch_first=True)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers, encoder_norm)

        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, d_feedforward, dropout, batch_first=True)
        decoder_norm = nn.LayerNorm(d_model)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, n_layers, decoder_norm)

        self.out_linear = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

        self.d_model = d_model
        self.n_head = n_head
    
    def _init_weights(self):
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        self.out_linear.bias.data.zero_()
        self.out_linear.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, 
                src,
                tgt,
                src_mask=None,
                tgt_mask=None,
                memory_mask=None,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        src = self.src_embedding(src)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)

        tgt = self.tgt_embedding(tgt)
        tgt = self.pos_encoder(tgt)
        
        tgt = self.transformer_decoder(tgt,
                                       memory,
                                       tgt_mask=tgt_mask,
                                       memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask,
                                       memory_key_padding_mask=memory_key_padding_mask)
        tgt = self.out_linear(tgt)

        return tgt
    

def generate_square_subsequent_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)