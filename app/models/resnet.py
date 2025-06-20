import torch
import torch.nn as nn
import torchvision.models as models

# resnet model, using resnet18 cause its not too heavy
class resnet_model(nn.Module):
    def __init__(self, num_classes):
        super(resnet_model, self).__init__()
        self.resnet = models.resnet18(weights=None)  # no pretrained, will load my weights
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)  # change final layer

    def forward(self, x):
        return self.resnet(x)  # just pass through resnet