from typing import List

import torch
import torch.nn as nn

OCTUPLE_MAX_IDXS = (255, 127, 128, 255, 127, 31, 253, 48)

class OctupleEmbedding(nn.Module):
    def __init__(self,
                 d_embed: int=64):
        super(OctupleEmbedding, self).__init__()
        self.d_embed = d_embed
        # +1(max_idx + 1), +1(<PAD>) +1(</s>), +1(<s>)
        self.linears = nn.ModuleList([
            nn.Embedding(d+1+1+1+1, self.d_embed) for d in OCTUPLE_MAX_IDXS
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for i in range(8):
            out.append(self.linears[i](x[:, i]))
        # オクタプルを連結 -> 64*8 = 512次元に
        out = torch.cat(out, 2)
        # 平均したい場合はこっち 加算等も同様
        # out = out.mean(dim=0)
        return out
    
class OctupleClassifier(nn.Module):
    def __init__(self,
                 d_model: int=512):
        super(OctupleClassifier, self).__init__()
        self.d_model = d_model
        self.linears = nn.ModuleList([
            nn.Linear(self.d_model, d+1+1+1+1) for d in OCTUPLE_MAX_IDXS
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = [linear(x) for linear in self.linears]
        return out
        