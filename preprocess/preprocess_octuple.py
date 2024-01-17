import glob
import os
from pathlib import Path

from tqdm import tqdm
import pickle

def make_split_tunes(octuple_path: str, out_dir: str='lmd_matched_octuple_split'):
    new_dir = out_dir + '/' + Path(octuple_path).name.replace('.txt', '')
    new_dir_pickle = new_dir.replace('split', 'pickle')
    if os.path.isdir(new_dir):
        return
    try:
        with open(octuple_path, 'r') as f:
            octuples = f.readlines()
    except OSError:
        with open('error_list.txt', 'a') as f:
            f.write(octuple_path + '\n')
        print(octuple_path)
        return 
    os.mkdir(new_dir)
    os.mkdir(new_dir_pickle)
    for i, oct in enumerate(octuples):
        output_path = new_dir_pickle + f'/{i}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(oct, f)
        output_path = new_dir + f'/{i}.txt'
        with open(output_path, 'w') as f:
            f.write(oct)

def make_split_dataset(src_dir: str='lmd_matched_octuple', tgt_dir: str='lmd_matched_octuple_split'):
    tune_paths = glob.glob(src_dir+'/*.txt')
    for tune_path in tqdm(tune_paths):
        make_split_tunes(tune_path)

if __name__ == "__main__":
    make_split_dataset()