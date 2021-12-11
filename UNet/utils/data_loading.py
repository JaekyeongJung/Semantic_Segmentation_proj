import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import json
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, cls_dir:str, scale: float = 1.0):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.cls_dir = Path(cls_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255
        else:
            img_ndarray = np.where(img_ndarray==10, 0, img_ndarray)
            img_ndarray = np.where(img_ndarray==20, 1, img_ndarray)
            img_ndarray = np.where(img_ndarray==30, 2, img_ndarray)
            img_ndarray = np.where(img_ndarray==40, 3, img_ndarray)
            img_ndarray = np.where(img_ndarray==50, 4, img_ndarray)
            img_ndarray = np.where(img_ndarray==60, 5, img_ndarray)
            img_ndarray = np.where(img_ndarray==70, 6, img_ndarray)
            img_ndarray = np.where(img_ndarray==80, 7, img_ndarray)
            img_ndarray = np.where(img_ndarray==100, 8, img_ndarray)

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        cls_file = list(self.cls_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(cls_file) == 1, f'Either no json found for the ID {name}: {cls_file}'
        
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        # b = mask[mask > 9]
        # if len(b) > 0:
        #     print(name)
            # print(b)
        
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


