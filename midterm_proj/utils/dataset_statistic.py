import os
from glob import glob
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class LandcoverDataset(Dataset):
    def __init__(self):
        self.root_dir = 'C:/Users/IIPL/Downloads/Semantic_Segmentation_proj/data/landcover_map_airplane_Gangwon/train/images'
        self.images_list = glob(self.root_dir + '/*.tif')
        
    def __getitem__(self, index):
        image = Image.open(self.images_list[index])
        image = transforms.ToTensor()(image)
        
        return image

    def __len__(self):
        return len(self.images_list)
    
if __name__ == '__main__':
    device = torch.device('cuda')
    dataset = LandcoverDataset()
    loader = DataLoader(dataset, batch_size=64, num_workers=1, shuffle=False)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in tqdm(loader):
        data = data.to(device)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    
    print("mean : ", mean)
    print("std : ", std)
    # mean :  [0.3420, 0.4124, 0.3591]
    # std :  [0.1731, 0.1451, 0.1206]