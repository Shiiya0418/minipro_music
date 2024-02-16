import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.octuple_linears import OctupleEmbedding, OctupleClassifier, OCTUPLE_MAX_IDXS
from models.octuple_transformer_decoder import OctupleTransformerDecoder

class OctupleTransformer(nn.Module):
    def __init__(self,
                 device=torch.device('cuda'),
                 d_embed: int=64,
                 d_model: int=64*8,
                 d_feedforward: int=2048,
                 n_head: int=8,
                 n_layers: int=6,
                 dropout: float=0.5):
        super(OctupleTransformer, self).__init__()
        self.device = device
        self.octuple_embedding = OctupleEmbedding(d_embed=d_embed)
        self.transformer_decoder = OctupleTransformerDecoder(
            device=device, d_model=d_model, d_feedforward=d_feedforward, n_head=n_head, n_layers=n_layers, dropout=dropout
        )
        self.octuple_classifier = OctupleClassifier(d_model=d_model)
    
    def forward(self, x: torch.Tensor,) -> List[torch.Tensor]:
        # x: [BATCH_SIZE, max_len, d_model] の形で入ってくる
        x = self.octuple_embedding(x)
        x = self.transformer_decoder(x)
        octuples = self.octuple_classifier(x)
        # List of [BATCH_SIZE, octuple_vocab_size]
        # よって、[[BATCH, 255], [BATCH, 127], ..., [BATCH, 48]]
        # みたいな感じ
        return octuples
    
    def greedy_decode(self,
                      max_len: int=1002,
                      start_symbol: int=0,
                      end_symbol: int=1):
        # batch_size = 1に限定する
        self.eval()
        with torch.no_grad():
            # [batch_size, 8(octuple), 1(length)]
            generated = torch.ones((1, 8, 1)).fill_(start_symbol).to(device=self.device, dtype=torch.int64)
            for i in range(max_len - 1):
                out = self(generated)
                octuple = []
                # [8, batch_size(1), length, vocab_size]
                for o in out:
                    # [batch_size(1), length]
                    o = o.argmax(dim=2)
                    # [batch_size(1), 1] 末尾のデータ（次のトークン）を獲得
                    next_word = o.data[:, -1]
                    # octupleのリストに格納
                    octuple.append(next_word)
                # [batch_size(1), 8, 1]
                octuple = torch.stack(octuple).unsqueeze(0)
                generated = torch.cat((generated, octuple), dim=2)
                if end_symbol in octuple or 2 in octuple:
                    break
            for j in range(8):
                generated[0][j][-1] = end_symbol
        # [batch_size(1), 8, length]
        return generated
                
            

