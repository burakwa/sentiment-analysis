import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from .config import Config
from .models import BaseEmotionModel

class Trainer:
    def __init__(self, model: BaseEmotionModel, train_dataset, val_dataset, config: Config):
        
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * images.size(0)
        return total_loss / len(dataloader.dataset)