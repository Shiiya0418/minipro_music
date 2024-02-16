import os
import warnings
from typing import Tuple, List, Dict

import torch
from torch.utils.data import DataLoader
import itertools
from miditoolkit import MidiFile
warnings.simplefilter('ignore')

import preprocess.midi_to_octuple as mto

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
            print(oct_note)
            notes.append(mto.str_to_encoding(oct_note))
        except AssertionError:
            print(f'{oct_note}: AssersionError')
    print(f'notes {len(notes)}')
    octuple_tune = list(itertools.chain.from_iterable(notes))
    print(f'octuple_tune {len(octuple_tune)}')
    midi_tune = mto.encoding_to_MIDI(octuple_tune)
    return midi_tune