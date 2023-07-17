# initialize model

import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict

class Model:
    model = torchvision.models.vgg16(pretrained=True)
    newClassifier = nn.Sequential(
    OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(p = 0.5)),
        ('fc2', nn.Linear(4096, 11)),
        ('output', nn.LogSoftmax(dim = 1))
        ])
    )
    model.classifier = newClassifier

    def __init__(self):
        self.load_model()

    def load_model(self):
        self.model.load_state_dict(torch.load("model\model_vgg16.pt", map_location=torch.device('cpu'))) #torch
        self.model.eval()

    @property
    def get_model(self):
        return self.model