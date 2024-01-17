from pathlib import Path
from copy import deepcopy

from miditok import REMI, TokenizerConfig, MIDITokenizer
from miditok.pytorch_data import DatasetTok, DataCollator
import pickle
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

def make_REMI(tokenizer_params: Dict=None):
    config = TokenizerConfig(**tokenizer_params) if tokenizer_params is not None else TokenizerConfig()
    config.use_programs = True
    tokens_path = Path('./lmd_matched_REMI')
    tokenizer = REMI(config)
    midi_paths = list(Path('./lmd_matched_flat').glob('**/*.mid'))
    tokenizer.tokenize_midi_dataset(midi_paths, tokens_path)


def REMIjson_to_midi(src_dir: str="lmd_matched_REMI", tgt_dir: str="lmd_matched_REMI_midi", tokenizer_params: Dict=None):
    config = TokenizerConfig(**tokenizer_params) if tokenizer_params is not None else TokenizerConfig()
    config.use_programs = True
    tokenizer = REMI(config)
    tokens_paths = list(Path(src_dir).glob('**/*.json'))
    tokens_paths = tokens_paths[:1]
    for tokens_path in tokens_paths:
        tokens = tokenizer.load_tokens(tokens_path)
        tokens = tokenizer(tokens['ids'])
        output_path = str(tokens_path).replace('./', '/').replace(src_dir, tgt_dir).replace('.json', '.mid')
        print(output_path)
        tokens.dump(output_path)
        # tokenizer.tokens_to_midi(tokens, output_path=output_path)

def preprocess_remi(src_dir:str='lmd_matched_REMI_elim', tgt_dir: str='lmd_matched_REMI_pickle', tokenizer_params: Dict=None):
    config = TokenizerConfig(**tokenizer_params) if tokenizer_params is not None else TokenizerConfig()
    config.use_programs = True
    tokenizer = REMI(config)
    tokens_paths = list(Path(src_dir).glob('**/*.json'))
    for tokens_path in tqdm(tokens_paths):
        tokens = tokenizer.load_tokens(tokens_path)
        tokens = tokenizer(tokens['ids'])
        output_path = str(tokens_path).replace('./', '/').replace(src_dir, tgt_dir).replace('.json', '.pkl')
        with open(output_path) as f:
            pickle.dump(tokens, f)


if __name__ == "__main__":
    pass
    # make_REMI()

    # REMIjson_to_midi()


# def encode_REMIlike(path: str="./lmd_matched_flat/0a0ce238fb8c672549f77f3b692ebf32.mid"):
#     encoded = midi_preprocessor.encode_midi(path)
#     return encoded

# def decode_REMIlike(encoded: List[int], path: str="./lmd_matched_REMIlike_decoded/0a0ce238fb8c672549f77f3b692ebf32.mid"):
#     midi_preprocessor.decode_midi(encoded, path)

# decode_REMIlike(encode_REMIlike())