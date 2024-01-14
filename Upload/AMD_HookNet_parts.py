import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mid_channels=None, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(in_channels, out_channels, kernel_size),
        )

    def forward(self, x):
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.catconv = Conv_block(out_channels * 2, out_channels, kernel_size, mid_channels=out_channels // 2)

    def forward(self, x, y):
        x = self.up(x)
        x = self.conv(x)
        crop_size = int(y.shape[3] - x.shape[3]) / 2
        _, _, h, w = y.shape
        item_cropped = y[:, :, int(crop_size):h - int(crop_size), int(crop_size):w - int(crop_size)]
        x = torch.cat((x, item_cropped), dim=1)
        return self.catconv(x)


class T_Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters, i, kernel_size=3):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        self.conv = Conv_block(in_channels, out_channels, kernel_size, mid_channels=out_channels // 2)
        self.catconv = Conv_block(out_channels * 2, out_channels, kernel_size, mid_channels=out_channels // 2)

        ratio = 4
        if i == 'up1':
            self.Pre = nn.Conv2d(64 * ratio, 4, kernel_size=1)
            self.queryd = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.keyd = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.valued = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.softmaxd = nn.Softmax(dim=-1)
            self.gammad = nn.Parameter(torch.zeros(1))

        elif i == 'up2':
            self.Pre = nn.Conv2d(32 * ratio, 4, kernel_size=1)
            self.queryd = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.keyd = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.valued = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.softmaxd = nn.Softmax(dim=-1)
            self.gammad = nn.Parameter(torch.zeros(1))

        elif i == 'up3':
            self.Pre = nn.Conv2d(16 * ratio, 4, kernel_size=1)
            self.queryd = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.keyd = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.valued = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.softmaxd = nn.Softmax(dim=-1)
            self.gammad = nn.Parameter(torch.zeros(1))

    def forward(self, x, y, z):
        crop_size = int(z.shape[3] - x.shape[3]) / 2
        _, _, h, w = z.shape
        item_cropped = z[:, :, int(crop_size):h - int(crop_size), int(crop_size):w - int(crop_size)]
        x = torch.cat((x, item_cropped), dim=1)
        B, C, H, W = x.shape
        xx = self.queryd(x).view(B, -1, H * W).permute(0, 2, 1)
        xxx = self.keyd(x).view(B, -1, H * W)
        _, head_dim, _ = xxx.shape
        energy = torch.bmm(xx, xxx) / (head_dim ** 0.5)
        attention = self.softmaxd(energy)
        value = self.valued(x).view(B, C, H * W)
        out = torch.bmm(value, attention).view(B, C, H, W)
        x = self.gammad * out + x
        x = self.up(x)
        x = self.conv(x)

        crop_size = int(y.shape[3] - x.shape[3]) / 2
        _, _, h, w = y.shape
        item_cropped = y[:, :, int(crop_size):h - int(crop_size), int(crop_size):w - int(crop_size)]
        x = torch.cat((x, item_cropped), dim=1)
        output = self.catconv(x)
        return output, self.Pre(output)


class Output(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()
        self.output = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x):
        return self.output(x)
