from typing import List

from torch.utils.data import Dataset, DataLoader

from ..preprocess.octuple_process import *

class OctupleDataset(Dataset):
    def __init__(self, dataset_paths: List[str], dict_path: str='dict.txt'):
        super.__init__(OctupleDataset)
        self.dataset_paths = dataset_paths
        self.get_octuple_dict(dict_path)

    def get_octuple_dict(self, dict_path: str='dict.txt'):
        with open(dict_path, 'r') as f:
            octuples = f.readlines()
        self.id_to_octuple = octuples
        self.octuple_to_id = {octuple: i for i, octuple in enumerate(self.id_to_octuple)}

    def __getitem__(self, index: int) -> List[int]:
        with open(self.dataset_paths[index], 'r') as f:
            item_str = f.readline()
        item_splited = item_str.split()
        return [self.octuple_to_id(item) for item in item_splited]

    def __len__(self) -> int:
        return len(self.dataset_paths)
    
