
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset

dir_img = Path('/Users/sungyoon-kim/Documents/GitHub/Sementic_segmentation_study/data/images/')
dir_mask = Path('/Users/sungyoon-kim/Documents/GitHub/Sementic_segmentation_study/data/masks/')
dir_label = Path('/Users/sungyoon-kim/Documents/GitHub/Sementic_segmentation_study/data/label_cls/')

if __name__ == '__main__':
    loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
    dataset = BasicDataset(dir_img, dir_mask, dir_label, scale=0.5)
    loader = DataLoader(dataset, shuffle=False, **loader_args)
    
    for batch in tqdm(loader):
        images = batch['image']
        true_masks = batch['mask']
        