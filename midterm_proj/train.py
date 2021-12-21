import builtins
import sys
import argparse
import logging
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.LandcoverDataset import LandcoverDataset
from utils.dice_score import dice_loss
from utils.utils import decode_segmap
from evaluate import evaluate
from models import UNet, VGG_UNet

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

root_dir = 'C:\\Users\\IIPL\\Downloads\\Semantic_Segmentation_proj\\data\\landcover_map_airplane_Gangwon'

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_size: tuple = (256,256),
              amp: bool = False):
    
    # 1. Create dataset
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    
    train_set = LandcoverDataset(root_dir, img_size=img_size, split='train')
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    n_train = train_set.__len__()
    
    val_set = LandcoverDataset(root_dir, img_size=img_size, split='val')
    val_loader = DataLoader(train_set, shuffle=True, **loader_args)
    n_val = val_set.__len__()

    # (Initialize logging)
    experiment = wandb.init(project='Semantic_Segmentation_project', entity='jaekyeong', name='VGG-UNet')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_size=img_size,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images size:     {img_size}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device, dtype=torch.long)
                
                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    softmax_ = F.softmax(masks_pred, dim=1).float()
                    one_hot_ = F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float()
                    loss = criterion(masks_pred, true_masks) + dice_loss(softmax_, one_hot_, multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
                # Evaluation round
                division_step = (n_train // (2 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)
                        
                        segmap = torch.softmax(masks_pred, dim=1)[0].detach().cpu().numpy()
                        segmap = torch.tensor(decode_segmap(segmap)).float()
                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(segmap)
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            torch.save(net.state_dict(),  f'checkpoints/VGG_U-Net_{epoch + 1:02d}.pt')
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--size', '-s', type=tuple, default=(256, 256), help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = VGG_UNet(n_channels=3, n_classes=9, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_size=args.size,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'checkpoints/INTERRUPTED.pt')
        logging.info('Saved interrupt')
        sys.exit(0)