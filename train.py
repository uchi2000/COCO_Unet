import os
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from UNet import UNet
from data_loading import BasicDataset

dir_img = Path('/workspace/COCO_segmentation/Data/image/train')
dir_mask = Path('/workspace/COCO_segmentation/Data/mask/train')
dir_checkpoint = Path('/workspace/COCO_segmentation/checkpoints')

def train_model(
    model,
    device,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    image_size: int = 256,
    save_checkpoint: bool = True
):
    dataset = BasicDataset(dir_img, dir_mask)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=True)
    
    experiment = wandb.init(project='COCO')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            image_size=image_size , save_checkpoint=save_checkpoint)
    )
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(dataset)}
        Image size:      {image_size}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
    ''')
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tqdm(data_loader, total = len(dataset) // batch_size, desc=f'Epoch {epoch}/{epochs}', unit='batch'):
            images, masks = batch['image'], batch['mask']
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks = masks.to(device=device, dtype=torch.float32)
            masks = masks.unsqueeze(1)
            optimizer.zero_grad()
            masks_pred = model(images)
            criterion_loss = criterion(masks_pred, masks)
            
            criterion_loss.backward()
            optimizer.step()
            experiment.log({
                'criterion loss': criterion_loss.item(),
                'epoch': epoch
            })
        
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
            
            
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    model = UNet(n_channels=3, n_classes=1)
    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)
    
    train_model(
            model=model,
            epochs=5,
            batch_size=16,
            learning_rate=1e-5,
            device=device
    )