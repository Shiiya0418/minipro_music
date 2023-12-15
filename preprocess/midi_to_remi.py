from pathlib import Path
from copy import deepcopy

from miditok import REMI, TokenizerConfig, Octuple
from miditok.pytorch_data import DatasetTok, DataCollator
from tqdm import tqdm

from typing import Dict, List

def make_REMI(tokenizer_params: Dict=None):
    config = TokenizerConfig(**tokenizer_params) if tokenizer_params is not None else TokenizerConfig()
    tokens_path = Path('./lmd_REMI_no_bpe')
    tokenizer = REMI(config)
    midi_paths = list(Path('./lmd_matched_flat').glob('**/*.mid'))
    tokenizer.tokenize_midi_dataset(midi_paths, tokens_path)

# def make_Octuple(tokenizer_params: Dict=None):
#     config = TokenizerConfig(**tokenizer_params) if tokenizer_params is not None else TokenizerConfig()
#     tokens_path = Path('./lmd_Octuple_no_bpe')
#     tokenizer = Octuple(config)
#     midi_paths = list(Path('./lmd_matched_flat').glob('**/*.mid'))
#     tokenizer.tokenize_midi_dataset(midi_paths, tokens_path)

make_REMI()