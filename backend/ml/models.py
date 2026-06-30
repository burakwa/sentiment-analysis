import torch.nn as nn
from torchvision import models

class BaseEmotionModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x): raise NotImplementedError
    
class ResNetEmotionModel(BaseEmotionModel):
    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)