import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.utils import decode_segmap

from dataset import LandcoverDataset
from models import UNet, VGG_UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                img_size=(256,256),
                out_threshold=0.5):
    net.eval()
    img_size = img_size
    mean = (0.3420, 0.4124, 0.3591)
    std = (0.1731, 0.1451, 0.1206)
    img_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Normalize(mean, std),
    ])
    img = img_tf(full_img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        # probs = torch.sigmoid(output)[0]

        segmap = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
        segmap = torch.tensor(decode_segmap(segmap)).float()
        
        mask_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        full_mask = mask_tf(segmap.cpu()).squeeze()
        print(full_mask)
        print(full_mask.shape)
        

    return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='checkpoints/VGG_U-Net_01.pt', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    in_files = glob('C:/Users/IIPL/Downloads/Semantic_Segmentation_proj/input/*.tif')
    out_files = 'C:/Users/IIPL/Downloads/Semantic_Segmentation_proj/output/midterm'

    net = VGG_UNet(n_channels=3, n_classes=9)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    for i, file_path in enumerate(in_files):
        filename = os.path.basename(file_path)
        logging.info(f'\nPredicting image {filename} ...')
        img = transforms.PILToTensor()(Image.open(file_path)).float()

        mask = predict_img(net=net,
                           full_img=img,
                           img_size=(256,256),
                           out_threshold=args.mask_threshold,
                           device=device)
        
        result = mask_to_image(mask)
        
        out_filename = os.path.join(out_files, filename)
        result.save(out_filename)
        logging.info(f'Mask saved to {out_filename}')
        
