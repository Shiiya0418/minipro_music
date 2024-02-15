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

