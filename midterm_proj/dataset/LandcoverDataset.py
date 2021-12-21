from os.path import join
from glob import glob
import numpy as np
from PIL import Image
import json
from tifffile import imread
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# mean :  [0.3420, 0.4124, 0.3591]
# std :  [0.1731, 0.1451, 0.1206]
class LandcoverDataset(Dataset):
    def __init__(self, root_dir: str, img_size=(256, 256), mean=(0.3420, 0.4124, 0.3591), std=(0.1731, 0.1451, 0.1206), split='train'):
        assert split == 'train' or split == 'val', 'split must be train or val'
        adds_path = join(root_dir, split)
        
        self.images_list = glob(join(adds_path, 'images') + '/*.tif')
        self.masks_list = glob(join(adds_path, 'masks') + '/*.tif')
        # self.labels_list = glob(join(adds_path, 'labels') + '/*.json')
        
        self.interpolation = transforms.functional.InterpolationMode('nearest')
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.transforms = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.Normalize(self.mean, self.std),
        ])
        
    def __len__(self):
        return len(self.images_list)
        # return 17
    
    @classmethod
    def preprocess(self, mask):
        labels = [10, 20, 30, 40, 50, 60, 70, 80, 100]
        labeled_mask = np.zeros((mask.shape))
        for i, label in enumerate(labels):
            labeled_mask = np.where(mask==label, i, labeled_mask)
        
        return labeled_mask
        

    def __getitem__(self, idx):
        image_file = transforms.PILToTensor()(Image.open(self.images_list[idx])).float()
        image_file = self.transforms(image_file)
        
        mask_file = imread(self.masks_list[idx])
        mask_file = torch.Tensor(self.preprocess(mask_file))
        mask_file.unsqueeze_(0)
        if mask_file.max() > 8 or mask_file.min() < 0:
            print("before resize : ", self.masks_list[idx])
        
        mask_file = transforms.Resize(self.img_size, interpolation=self.interpolation)(mask_file).long()
        mask_file.squeeze_()
        
        # label_file = json.load(open(self.labels_list[idx]))
        
        if mask_file.max() > 8 or mask_file.min() < 0:
            print("after resize : ", self.masks_list[idx])
        
        return {
            'image': image_file,
            'mask': mask_file
        }

if __name__ == '__main__':
    dataset = LandcoverDataset(root_dir='C:\\Users\\IIPL\\Downloads\\Semantic_Segmentation_proj\\data\\landcover_map_airplane_Gangwon')
    print("Dataset Length : ", dataset.__len__())
    
    loader_args = dict(batch_size=32, num_workers=0, pin_memory=True)
    loader = DataLoader(dataset, shuffle=False, **loader_args)
    i = 0
    for batch in tqdm(loader):
        i += 1
        
        
        
    # data = dataset.__getitem__(1)
    # print(data['image'].shape, data['image'].dtype)
    # print(data['mask'].shape, data['mask'].dtype)
    # print(data['mask'])

