import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.octuple_transformer import OctupleTransformer
from octuple_dataset import OctupleDataset, collate_fn

if __name__ == "__main__":
    BATCH_SIZE = 2
    LERNING_RATE = 5e-4
    DEVICE = torch.device('cuda')
    EPOCHS = 10
    
    model = OctupleTransformer()
    print(model)
    file_list = glob.glob('lmd_matched_octuple_pickle/0a0dadc29a6ad3bec746c8b5af9c41cd/*.pkl', recursive=True)
    dataset = OctupleDataset(file_list)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, drop_last=True)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.octuple_to_id['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=LERNING_RATE)
    criterion = criterion.to(DEVICE)
    model = model.to(DEVICE)
    
    for epoch in tqdm(range(EPOCHS)):
        epoch_loss = 0
        for x, t in dataloader:
            x, t = x.to(DEVICE), t.to(DEVICE)
            print(f'x: {x.shape}')
            print(f't: {t.shape}')
            y = model(x)
            print(f'y: {len(y)}')
            loss = 0
            for i, (y_oct, _t) in enumerate(zip(y, t.permute(1, 0, 2))):
                print(f'y{i}: {y_oct.shape}')
                y_oct, _t = y_oct.reshape(y_oct.shape[0]*y_oct.shape[1], y_oct.shape[2]), _t.reshape(_t.shape[0]*_t.shape[1])
                print(f'y_oct: {y_oct.shape} t: {_t.shape}')
                loss += criterion(y_oct, _t)
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        break