import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.layer_1 = nn.Linear(28 * 28, 512)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(512, 10)
        self.final_activation = nn.Sigmoid()
        pass
        # Define the architecture here
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        images = self.layer_1(images)
        images = self.activation(images)
        images = self.dropout(images)
        images = self.classifier(images)
        images = self.final_activation(images)

        return torch.round(images, decimals=4)
        pass
        # Return the model's prediction to 4 decimal places
