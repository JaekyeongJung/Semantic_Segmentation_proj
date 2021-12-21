from .unet_parts import *
from torchvision.models import vgg16

class VGG_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(VGG_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        
        # Encoder
        self.backbone = vgg16(pretrained=True).features
        
        self.inc = self.backbone[:4]        # torch.Size([B, 64, 256, 256])
        self.down1 = self.backbone[4:9]     # torch.Size([B, 128, 128, 128])
        self.down2 = self.backbone[9:16]    # torch.Size([B, 256, 64, 64])
        self.down3 = self.backbone[16:23]   #  torch.Size([B, 512, 32, 32])
        self.down4 = self.backbone[23:30]   # torch.Size([B, 512, 16, 16])
        
        # Decoder
        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print("x1 : ", x1.shape)
        x2 = self.down1(x1)
        # print("x2 : ", x2.shape)
        x3 = self.down2(x2)
        # print("x3 : ", x3.shape)
        x4 = self.down3(x3)
        # print("x4 : ", x4.shape)
        x5 = self.down4(x4)
        # print("x5 : ", x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
