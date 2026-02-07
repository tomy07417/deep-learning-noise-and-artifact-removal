import torch
import numpy as np
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class Conv3K(torch.nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()      
        self.conv1 = torch.nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.conv1(x)

class DoubleConv(torch.nn.Module):
    '''
    '''
    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.double_conv = torch.nn.Sequential(
            Conv3K(channels_in, channels_out),
            torch.nn.BatchNorm2d(channels_out),
            torch.nn.ReLU(),

            Conv3K(channels_out, channels_out),
            torch.nn.BatchNorm2d(channels_out),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)

class DownConv(torch.nn.Module):
    '''
    '''
    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.MaxPool2d(2,2),
            DoubleConv(channels_in, channels_out),
        )

    def forward(self, x):
        return self.encoder(x)
    
class UpConv(torch.nn.Module):
    '''
    '''
    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.upsample = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bicubic'),
            torch.nn.Conv2d(channels_in, channels_in // 2, kernel_size=1, stride=1),
        )

        self.decoder = DoubleConv(channels_in, channels_out)
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x1, x2], dim=1)
        
        return self.decoder(x)
    
class UNet(torch.nn.Module):
    '''
    '''
    def __init__(self, channels_in, channels):
        super().__init__()

        self.initial_conv = DoubleConv(channels_in, channels)
        self.down_conv1 = DownConv(channels, channels * 2)
        self.down_conv2 = DownConv(channels * 2, channels * 4)
        self.down_conv3 = DownConv(channels * 4, channels * 8)
        self.down_conv4 = DownConv(channels * 8, channels * 16)

        self.up_conv1 = UpConv(channels * 16, channels * 8)
        self.up_conv2 = UpConv(channels * 8, channels * 4)
        self.up_conv3 = UpConv(channels * 4, channels * 2)
        self.up_conv4 = UpConv(channels * 2, channels)

        self.final_conv = torch.nn.Conv2d(channels, channels_in, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.initial_conv(x)
        x2 = self.down_conv1(x1)
        x3 = self.down_conv2(x2)
        x4 = self.down_conv3(x3)
        x5 = self.down_conv4(x4)

        u1 = self.up_conv1(x5, x4)
        u2 = self.up_conv2(u1, x3)
        u3 = self.up_conv3(u2, x2)
        u4 = self.up_conv4(u3, x1)

        return self.final_conv(u4)


def temporal_smooth(video_frames, strength=0.6):
    """
    video_frames: array shape (T, H, W)
    strength: peso del frame central (0.5–0.8 recomendado)
    """
    smoothed = video_frames.copy().astype(np.float32)
    
    for t in range(1, len(video_frames) - 1):
        smoothed[t] = (
            strength * video_frames[t] +
            (1 - strength) / 2 * video_frames[t-1] +
            (1 - strength) / 2 * video_frames[t+1]
        )
    
    return np.clip(smoothed, 0, 255).astype(np.uint8)

def test():
    x = torch.randn((32, 3, 256, 256))
    model = UNet(3, 64)
    return model(x)

pred = test()