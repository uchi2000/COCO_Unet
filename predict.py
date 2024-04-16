import os
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from os import listdir
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from UNet import UNet

model_path = Path('./checkpoints/checkpoint_epoch5.pth')
img_dir = Path('/workspace/COCO_segmentation/Data/image/val')
mask_dir = Path('/workspace/COCO_segmentation/Data/mask/val')

@torch.no_grad()
def predict(
    model,
    device
):
    model.eval()
    ids = [file.split('.')[0] for file in listdir(img_dir)]
    criterion = nn.BCEWithLogitsLoss()
    total_pred_score = 0
    
    for id in tqdm(ids, total=len(ids), desc='prediction', unit='image'):
        mask_file = list(mask_dir.glob(id + '.*'))
        img_file = list(img_dir.glob(id + '.*'))
        mask = Image.open(mask_file[0]).convert('L')
        img = Image.open(img_file[0]).convert('RGB')
        w, h = img.size
        mask = mask.resize((256, 256))
        img = img.resize((256, 256))
        
        img = np.asarray(img) / 255.0
        mask = np.asarray(mask) / 255.0
        image = torch.as_tensor(img).float().contiguous()
        mask = torch.as_tensor(mask).float().contiguous()
        image = image.to(device=device, dtype=torch.float32)
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
        mask_pred = model(image)
        mask_pred = mask_pred.squeeze(0).squeeze(0)
        criterion_loss = criterion(mask_pred, mask)
        total_pred_score += criterion_loss
        
        img = Image.fromarray((img * 255).astype(np.uint8))
        img = img.resize((w,h))
        img.save(os.path.join(f'/workspace/COCO_segmentation/pred/{id}_image.png'))
        mask_pred = torch.sigmoid(mask_pred) > 0.5
        mask_pred_image = mask_pred.cpu().numpy()
        mask_pred_image = Image.fromarray((mask_pred_image * 255).astype(np.uint8))
        mask_pred_image = mask_pred_image.resize((w,h))
        mask_pred_image.save(os.path.join(f'/workspace/COCO_segmentation/pred/{id}_mask_pred.png'))
        
    print( total_pred_score / len(ids) )      


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    model = UNet(n_channels=3, n_classes=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model_path}')
    logging.info(f'Using device {device}')
    model.to(device=device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    logging.info('Model loaded!')
    
    predict(
        model=model,
        device=device
    )