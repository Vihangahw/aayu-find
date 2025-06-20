import torch
import torch.nn as nn

# unet for segmenting leaves, fixed layer names to match my saved weights
class my_unet(nn.Module):
    def __init__(self, num_classes=1):  # binary mask, so 1 class
        super(my_unet, self).__init__()
        # encoder stuff
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # first conv
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # pooling for downsampling
        self.relu = nn.ReLU()
        # decoder stuff, upsampling and concat
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # concat with enc3
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)  # final mask output

    def forward(self, x):
        # encoder path, save for skip connections
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(self.pool(e1)))
        e3 = self.relu(self.enc3(self.pool(e2)))
        e4 = self.relu(self.enc4(self.pool(e3)))
        # decoder with skip connections
        d1 = self.upconv1(e4)
        d1 = torch.cat([d1, e3], dim=1)  # concat with e3
        d1 = self.relu(self.dec1(d1))
        d2 = self.upconv2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.relu(self.dec2(d2))
        d3 = self.upconv3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.relu(self.dec3(d3))
        out = self.final_conv(d3)  # binary mask
        return out