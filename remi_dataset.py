from pathlib import Path
from copy import deepcopy

from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetTok, DataCollator
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from typing import Dict, List

TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": True,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": False,
    "use_programs": False,
    "num_tempos": 32,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
}

def get_REMI_dataset(src_dir: str='lmd_matched_REMI_elim', tokenizer_params: Dict=None):
    config = TokenizerConfig(**tokenizer_params) if tokenizer_params is not None else TokenizerConfig()
    config.use_programs = True
    tokenizer = REMI(config)
    tokens_paths = list(Path(src_dir).glob('**/*.json'))
    remi_dataset = DatasetTok(tokens_paths, 0, 1000, tokenizer)
    return remi_dataset
