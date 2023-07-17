# this file is for handling all of the preprocssing

from PIL import Image
from torchvision.transforms import transforms
import torch
import numpy as np
import base64
from io import BytesIO


class Preprocessing:
    device = torch.device("cpu") 
    image_transforms_test = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(), # For GPU purpose
    ])

    def __init__(self, img_name):
        self.img_name = img_name

    def load_image(self):
        
        image = Image.open(BytesIO(base64.b64decode(self.img_name)))
        # image = Image.open(self.img_name).convert('RGB')
        image = self.image_transforms_test(image).float()
        image = image.to(self.device)
        self.image = image.unsqueeze(0)
        # return self.image

    @property
    def real_image(self):
        return self.img_name

    @property
    def get_image_(self):
        return self.image
