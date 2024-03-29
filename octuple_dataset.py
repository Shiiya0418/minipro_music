import pickle
import sys
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

# from preprocess.octuple_process import *

# すべての属性に<s>, </s>, <pad> を入れるので、3つ分足したうえで、各属性値のスタートidxを計算する
START_IDXS = torch.Tensor([-3, 259-3, 387-3, 516-3, 772-3, 900-3, 932-3, 1186-3])
PAD_NUM = 2

class OctupleDataset(Dataset):
    def __init__(self, dataset_paths: List[str], dict_path: str='octuple_dict.txt'):
        super(OctupleDataset).__init__()
        self.dataset_paths = dataset_paths
        self.get_octuple_dict(dict_path)

    def get_octuple_dict(self, dict_path: str='octuple_dict.txt'):
        with open(dict_path, 'r') as f:
            octuples = ['<s>', '</s>', '<pad>'] + [e.replace('\n', '') for e in f.readlines()]
        self.id_to_octuple = octuples
        self.octuple_to_id = {octuple: i for i, octuple in enumerate(self.id_to_octuple)}

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with open(self.dataset_paths[index], 'rb') as f:
            item_str = pickle.load(f)
        # なんか最後の '</s>' が抜けてたので、補完
        item_splited = item_str.split() + ['</s>']
        # idに直してtorch.Tensorを作る -> 8つ組に整形 [2~1002, 8]
        oct_elem = torch.Tensor([self.octuple_to_id[item] for item in item_splited]).reshape(int(len(item_splited)/8), 8)
        # それぞれの属性の開始idxを引くことで、どの属性も0スタートにする（Embedding用）
        sdx = START_IDXS.repeat(oct_elem.shape[0], 1)
        # ただし、頭とケツはsなので引かない
        sdx[0] = torch.zeros(8)
        sdx[-1] = torch.zeros(8)
        octuple = oct_elem - sdx
        octuple = octuple.to(dtype=torch.int64)
        target = octuple[1:]
        return octuple, target

    def __len__(self) -> int:
        return len(self.dataset_paths)

def collate_fn(batch):
    # batch処理を行う時に、長さをそろえるためのやつ
    octuple, target = list(zip(*batch))
    octuple = torch.nn.utils.rnn.pad_sequence(octuple, batch_first=True, padding_value=2).permute(0, 2, 1)
    target = torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=2).permute(0, 2, 1)
    pad = torch.full((target.shape[0], target.shape[1], 1), fill_value=PAD_NUM)
    target = torch.cat((target, pad), dim=2)
    return octuple, target

if __name__ == "__main__":
    import glob
    # file_list = glob.glob('lmd_matched_octuple_pickle/**/*.pkl', recursive=True)
    file_list = glob.glob('lmd_matched_octuple_pickle/0a0dadc29a6ad3bec746c8b5af9c41cd/*.pkl', recursive=True)
    dataset = OctupleDataset(file_list)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    for x, t in dataloader:
        print(x)
        break