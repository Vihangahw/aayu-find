import torch
import torch.nn as nn
from app.models import resnet_model, leaf_cnn, effnet_model

class EnsembleModel(nn.Module):
    def __init__(self, num_classes):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList([
            resnet_model(num_classes),
            leaf_cnn(num_classes),
            effnet_model(num_classes)
        ])
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False
        self.fc = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        outputs = torch.stack([model(x) for model in self.models], dim=0)
        avg_output = torch.mean(outputs, dim=0)
        return self.fc(avg_output)