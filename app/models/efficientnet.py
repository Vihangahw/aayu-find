import torch
import torch.nn as nn
import torchvision.models as models

# efficientnet b0, smaller model but good performance
class effnet_model(nn.Module):
    def __init__(self, num_classes):
        super(effnet_model, self).__init__()
        self.effnet = models.efficientnet_b0(weights=None)  # load my weights later
        num_ftrs = self.effnet.classifier[1].in_features  # get final layer size
        self.effnet.classifier = nn.Linear(num_ftrs, num_classes)  # custom classifier

    def forward(self, x):
        return self.effnet(x)