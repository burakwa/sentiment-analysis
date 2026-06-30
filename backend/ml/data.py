import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from .config import Config

class EmotionDataModule:
    def __init__(self , config: Config):
        self.config = config
        self.transforms = self._get_transforms()
    def _get_transforms(self):
        return {
            "train": transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "val": transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
    def get_dataloaders(self) -> dict:
        train_dir = os.path.join(self.config.data_dir, "train")
        val_dir = os.path.join(self.config.data_dir, "val")
        
        train_ds = datasets.ImageFolder(train_dir, transform=self.transforms["train"])
        val_ds = datasets.ImageFolder(val_dir, transform=self.transforms["val"])
        
        return {
            "train": DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True),
            "val": DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)
        }