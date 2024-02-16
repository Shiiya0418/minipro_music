import glob
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.octuple_transformer import OctupleTransformer
from octuple_dataset import OctupleDataset, START_IDXS
import preprocess.octuple_to_midi as otm

def attr_to_octuple(tokens: torch.Tensor)->torch.Tensor:
    # input: [length, 8] (0~256)
    # output: [length, 8] (0~12??)
    sdx = START_IDXS.repeat(tokens.shape[0], 1)
    sdx[0] = torch.zeros(8)
    sdx[-1] = torch.zeros(8)
    octuple =tokens + sdx.to(tokens.device, dtype=torch.int64)
    return octuple

if __name__ == "__main__":
    DEVICE = torch.device('cuda')
    
    # model_path = "./model_best.pt"
    
    model = OctupleTransformer()
    # model.load_state_dict(torch.load(model_path))
    
    model = model.to(DEVICE)
    
    output = model.greedy_decode()
    # [length, 8] のオクタプルっぽい感じになっているはず
    output = output.squeeze().permute(1, 0)
    octuple = attr_to_octuple(output)
    print(octuple)
    print(octuple.shape)
    
    octuple_dataset = OctupleDataset([])
    
    midi_tune = otm.octuple_to_midi(octuple.tolist(), octuple_dataset.id_to_octuple)
    print(midi_tune)
    midi_tune.dump('sample.mid')
    
    
    