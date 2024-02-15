import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class OctupleTransformerDecoder(nn.Module):
    def __init__(self,
                 device=torch.device('cuda'),
                 d_model=512,
                 d_feedforward=2048,
                 n_head=8,
                 n_layers=6,
                 dropout=0.5):
        super(OctupleTransformerDecoder, self).__init__()
        self.device = device
        self.pos_encoder = PositionalEncoding(d_model)

        # decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, d_feedforward, dropout, batch_first=True)
        # decoder_norm = nn.LayerNorm(d_model)
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layer, n_layers, decoder_norm)
        self.transformer_blocks = nn.ModuleList([
            DecoderBlock(d_model, d_feedforward, n_head, dropout)
            for _ in range(n_layers)
        ])
        self.d_model = d_model
        self.n_head = n_head
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [BATCH_SIZE, max_len, d_model] の形で入ってくる
        x = self.pos_encoder(x)
        for transformer_block in self.transformer_blocks:
            target_mask = generate_square_subsequent_mask(x.size(1)).to(self.device)
            x = transformer_block(x, target_mask)
        return x
    

def generate_square_subsequent_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1002):
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

class DecoderBlock(nn.Module):
    def __init__(self,
                 d_model=512,
                 d_feedforward=2048,
                 n_head=8,
                 dropout=0.5):
        super(DecoderBlock, self).__init__()
        
        self.self_attention = nn.MultiheadAttention(d_model, n_head, dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.linear2 = nn.Linear(d_feedforward, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, target_mask):
        attn_output, _ = self.self_attention(x, x, x, attn_mask=target_mask)
        x = x + self.dropout1(attn_output)
        x = self.layer_norm1(x)
        linear_output = self.linear2(F.gelu(self.linear1(x)))
        x = x + self.dropout2(linear_output)
        x = self.layer_norm2(x)
        return x