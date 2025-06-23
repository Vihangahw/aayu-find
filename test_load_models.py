import torch
from app.models import my_unet, resnet_model, effnet_model, leaf_cnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"running on: {device}")

num_classes = 4

try:
    unet = my_unet().to(device)
    unet.load_state_dict(torch.load('models/unet/unet_best.pth', map_location=device, weights_only=True))
    unet.eval()
    print("unet loaded ok!!")
except Exception as e:
    print(f"unet failed: {e}")

try:
    resnet = resnet_model(num_classes).to(device)
    state_dict = torch.load('models/resnet/best_resnet.pth', map_location=device, weights_only=True)
    new_state_dict = {f'resnet.{k}': v for k, v in state_dict.items()}
    resnet.load_state_dict(new_state_dict)
    resnet.eval()
    print("resnet loaded, finally!")
except Exception as e:
    print(f"resnet broke: {e}")

try:
    effnet = effnet_model(num_classes).to(device)
    state_dict = torch.load('models/efficientnet/best_efficientnet.pth', map_location=device, weights_only=True)
    new_state_dict = {k.replace('model.', 'effnet.'): v for k, v in state_dict.items() if 'num_batches_tracked' not in k}
    effnet.load_state_dict(new_state_dict)
    effnet.eval()
    print("effnet loaded, please work this time!")
except Exception as e:
    print(f"effnet still busted: {e}")

try:
    cnn = leaf_cnn(num_classes).to(device)
    cnn.load_state_dict(torch.load('models/cnn/classifier_model2_best.pth', map_location=device, weights_only=True))
    cnn.eval()
    print("cnn loaded, all good")
except Exception as e:
    print(f"cnn failed: {e}")

try:
    # Define EnsembleModel inline to match the original architecture
    class EnsembleModel(torch.nn.Module):
        def __init__(self, num_classes):
            super(EnsembleModel, self).__init__()
            self.models = torch.nn.ModuleList([
                resnet_model(num_classes),
                leaf_cnn(num_classes),
                effnet_model(num_classes)
            ])
            for model in self.models:
                for param in model.parameters():
                    param.requires_grad = False
            self.fc = torch.nn.Linear(num_classes, num_classes)  # Matches the checkpoint's expected input size

        def forward(self, x):
            outputs = torch.stack([model(x) for model in self.models], dim=0)
            avg_output = torch.mean(outputs, dim=0)  # Average the outputs
            return self.fc(avg_output)

    ensemble = EnsembleModel(num_classes).to(device)
    state_dict = torch.load('models/ensemble/ensembleUNET_model_best.pth', map_location=device, weights_only=True)
    # Adjust state dict keys to match the new model structure
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('models.0.'):
            new_key = key.replace('models.0.', 'models.0.resnet.')
            new_state_dict[new_key] = value
        elif key.startswith('models.1.'):
            new_key = key.replace('models.1.', 'models.1.')
            new_state_dict[new_key] = value
        elif key.startswith('models.2.'):
            new_key = key.replace('models.2.', 'models.2.effnet.')
            new_state_dict[new_key] = value
        elif key == 'fc.weight':
            new_state_dict['fc.weight'] = value
        elif key == 'fc.bias':
            new_state_dict['fc.bias'] = value
    ensemble.load_state_dict(new_state_dict, strict=False)
    ensemble.eval()
    print("ensemble loaded, all good")
except Exception as e:
    print(f"ensemble failed: {e}")