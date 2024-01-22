import os
import warnings
from typing import Tuple, List, Dict

import torch
from torch.utils.data import DataLoader
import itertools
from miditoolkit import MidiFile
warnings.simplefilter('ignore')

import midi_to_octuple as preprocess

BOS_NUM = 0
BOS_VALUE = '<s>'
PAD_NUM = 1
PAD_VALUE = '<pad>'
MASK_NUM = 1236
MASK_VALUE = '<mask>'
EOS_NUM = 2
EOS_VALUE = '</s>'
UNK_NUM = 3
UNK_VALUE = '<unk>'
NUM_GENRE = 25

BATCH_SIZE = 32

def octuple_to_midi(
    octuple_tokens: List[str or List[str, str, str, str, str, str, str, str]],
    id_to_octuple: List[str],
) -> MidiFile:
    """octupleトークンのidのリストを受け取ります。[0, 0, 0, 0, 0, 0, 0, 0, 3, 135, 299, ..., 2, 2, 2, 2, ]みたいな

    Args:
        octuple_tokens (List[str or List[str, str, str, str, str, str, str, str]]): midiにしたいトークンリスト
        id_to_octuple (List[str]): トークナイズに使った辞書

    Returns:
        MidiFile: midi_file.dump(ファイル名) で、midiにできます。
    """
    if type(octuple_tokens[0]) == list:
        octuple_tokens = list(itertools.chain.from_iterable(octuple_tokens))
    octuple_tune = [id_to_octuple[id_oct] for id_oct in octuple_tokens]
    notes = []
    for i in range(0, len(octuple_tune), 8):
        oct_note = ' '.join(octuple_tune[i: i+8])
        try:
             notes.append(preprocess.str_to_encoding(oct_note))
        except AssertionError:
            print(f'{oct_note}: AssersionError')
    print(f'notes {len(notes)}')
    octuple_tune = list(itertools.chain.from_iterable(notes))
    print(f'octuple_tune {len(octuple_tune)}')
    midi_tune = preprocess.encoding_to_MIDI(octuple_tune)
    return midi_tune

def generate_greedy(
    model: MusicBART,
    dataloader: DataLoader,
    log_dir_path: str,
    n: int=20,
    max_len: int=1002,
    start_symbol: int=BOS_NUM,
    end_symbol: int=EOS_NUM,
    device: torch.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
) -> Tuple[List]:
    # batch_size = 1 限定
    # 検証フェーズ
    dataset = dataloader.dataset
    id2token_list = dataset.src_dict.symbols
    model.eval()
    with torch.no_grad():
        generated_songs = []
        source_songs = []
        for j, batch in enumerate(dataloader):
            source, _ = batch
            source = source.to(model.device)
            generated = model.greedy_decode(source, max_len, start_symbol, end_symbol)
            octuple_tune =[]
            source_tune = []
            # return generated
            # print("Generation")
            for note in generated[0]:
                octuple_tune.append(id2token_list[note])
            octuple_tune = ' '.join(octuple_tune)
            notes = []
            # print(octuple_tune.split()[0:0+8])
            for i in range(0, len(octuple_tune.split()), 8):
                oct_note = ' '.join(octuple_tune.split()[i: i+8])
                try:
                    notes.append(preprocess.str_to_encoding(oct_note))
                except AssertionError:
                    print(f'{oct_note}: AssersionError')
            print(f'notes {len(notes)}')
            octuple_tune = list(itertools.chain.from_iterable(notes))
            print(f'octuple_tune {len(octuple_tune)}')
            if len(octuple_tune) == 0:
                print(f'skip {j}')
                continue
            midi_tune = preprocess.encoding_to_MIDI(octuple_tune)
            generated_songs.append(midi_tune)
            # print("Source")
            for note in source[0]:
                source_tune.append(id2token_list[note])
            source_tune = ' '.join(source_tune)
            notes = []
            for i in range(0, len(source_tune.split()), 8):
                oct_note = ' '.join(source_tune.split()[i: i+8])
                try:
                    notes.append(preprocess.str_to_encoding(oct_note))
                except AssertionError:
                    print(f'{i}: AssersionError')
            source_tune = list(itertools.chain.from_iterable(notes))
            midi_tune = preprocess.encoding_to_MIDI(source_tune)
            source_songs.append(midi_tune)
            if j == n:
                break
    return generated_songs, source_songs