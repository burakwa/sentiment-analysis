from dataclasses import dataclass , field
from typing import List, Tuple

@dataclass
class Config:
    data_dir: str = "./datasets"
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-4
    image_size: Tuple[int, int] = (512, 512)
    num_classes: int = 6
    class_names: List[str] = field(default_factory=lambda: ["disgust", "fear", "sad" , "happy", "surprise" , "neutral"])
    device: str = "cuda"