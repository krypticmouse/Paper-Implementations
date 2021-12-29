import numpy as np
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

from config import *
from model import MSRNet
from loss import FrobiniusLoss
from dataset import LowLightDataset

data = LowLightDataset('/content/drive/MyDrive/Light Dataset/BrighteningTrain')
train_data, valid_data = random_split(data, [800,200])

trainloader = DataLoader(train_data, batch_size = BATCH_SIZE)
validloader = DataLoader(valid_data, batch_size = BATCH_SIZE)

model = MSRNet(N, K, V)
if torch.cuda.is_available():
    model = model.cuda()

optimizer = optim.AdamW(model.parameters(), 
                        lr = LEARNING_RATE, 
                        weight_decay = WEIGHT_DECAY)
criterion = FrobiniusLoss()

min_valid_loss = np.inf
for e in range(EPOCHS):
    model.train()

    train_loss = 0.0
    for batch in tqdm(trainloader):
        optimizer.zero_grad(set_to_none = True)
        high, low = batch['high'], batch['low']

        if torch.cuda.is_available():
            high, low = high.cuda(), low.cuda()

        high_pred = model(low)
        loss = criterion(high_pred, high)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    model.valid()

    valid_loss = 0.0
    for batch in tqdm(validloader):
        high, low = batch['high'], batch['low']
        if torch.cuda.is_available():
            high, low = high.cuda(), low.cuda()

        high_pred = model(low)
        loss = criterion(high_pred, high)
        valid_loss += loss.item()
    
    print(f'Epoch {e}\t\t\t Training Loss: {train_loss:%.4f}\t\t\t Validation Loss: {valid_loss:%.4f}')
    if valid_loss < min_valid_loss:
        print('Saving Model...')
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), 'msrnet.pth')