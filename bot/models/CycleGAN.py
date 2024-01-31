from PIL import Image
import os

import torch
from torchvision.transforms import v2
import torchvision.transforms as transforms
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_channels = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_channels, 7),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        ]
        in_channels = out_channels

        # Encoder
        for _ in range(2):
            out_channels *= 2
            model += [
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
            ]
            in_channels = out_channels

        # Residual blocks

        for _ in range(num_residual_blocks):
            model += [ResNetBlock(out_channels)]

        # Decoder
        for _ in range(2):
            out_channels //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
            ]
            in_channels = out_channels

        # Output layer
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_channels, channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class CycleGANClass:
    def __init__(self):
        self.model = GeneratorResNet((3,256,256), 9)
        self.model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'bot', 'models',
                                                           'pretrained_models','cycle.pth')))

    
    def forward(self, pic_path, main_path):

        im = Image.open(pic_path)
        w,h = im.size
    
        if h or w >= 500:
            scale_factor = 500/max(h,w)
            height = round(h * scale_factor)
            width = round(w * scale_factor)
            im = transforms.Resize(size = (height,width))(im)

        tp = transforms.PILToTensor()
        td = v2.ToDtype(torch.float32, scale=True)
        nm = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

        im = tp(im)
        im = td(im)
        input = nm(im)
        self.model.eval()

        output = self.model(input.unsqueeze(0))
        output = output/2 + 0.5
        gen_image = (transforms.ToPILImage()(output.squeeze(0)))
        
        gen_image.save(os.path.join(main_path, 'pic4.jpg'))
