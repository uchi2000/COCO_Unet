import numpy as np
import torch
from os import listdir
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)

        self.ids = [file.split('.')[0] for file in listdir(images_dir)]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img):
        newW, newH = 256, 256
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST)
        img = np.asarray(pil_img)
        return img / 255.0

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        
        if img.mode != 'RGB':
            img = img.convert('RGB')

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img)
        img = img.transpose((2, 0, 1))
        mask = self.preprocess(mask)
        
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }