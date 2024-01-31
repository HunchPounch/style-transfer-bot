import os
import PIL

import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

from bot.models.StyleTransfer_methods import run_style_transfer, image_loader


class StyleTransferClass:
    
    def __init__(self):
        self.cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])


    def forward(self, content_img_path, style_img_path, main_path):
        
        device = "cpu"
        imsize = 244
        unloader = transforms.ToPILImage()
        style_img, content_img, w, h  = image_loader(style_img_path,content_img_path, device, imsize)
        input_img = content_img.clone()

        output = run_style_transfer(self.cnn, self.cnn_normalization_mean, self.cnn_normalization_std,
                                     content_img, style_img, input_img)
        image = output.cpu().clone()  
        image = image.squeeze(0)      
        image = unloader(image)

        if w < h:
            h1 = round(imsize*1.2)
            w1 = round(imsize*(w/h)*1.2)
        else:
            w1 = round(imsize*1.2)
            h1 = round(imsize*(h/w)*1.2)

        image_sized = PIL.ImageOps.fit(image, (w1,h1))
        image_sized.save(os.path.join(main_path, 'pic3.jpg'))
 


