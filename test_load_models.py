import torch
from app.models import my_unet, resnet_model, effnet_model, leaf_cnn

# testing model loading, effnet still broken, ugh
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"running on: {device}")  # should be cuda now, fingers crossed

num_classes = 4  # 4 classes for all? need to double-check effnet

try:
    # unet good, no changes
    unet = my_unet().to(device)
    unet.load_state_dict(torch.load('models/unet/unet_best.pth', map_location=device, weights_only=True))
    print("unet loaded ok!!")
except Exception as e:
    print(f"unet failed, why now?: {e}")

try:
    # resnet worked, keeping it
    resnet = resnet_model(num_classes).to(device)
    state_dict = torch.load('models/resnet/best_resnet.pth', map_location=device, weights_only=True)
    new_state_dict = {f'resnet.{k}': v for k, v in state_dict.items()}
    resnet.load_state_dict(new_state_dict)
    print("resnet loaded, finally!")
except Exception as e:
    print(f"resnet broke again: {e}")

try:
    # effnet, weights have model. prefix, strip that, ignore num_batches_tracked
    effnet = effnet_model(num_classes).to(device)
    state_dict = torch.load('models/efficientnet/best_efficientnet.pth', map_location=device, weights_only=True)
    # strip model. and skip num_batches_tracked
    new_state_dict = {k.replace('model.', 'effnet.'): v for k, v in state_dict.items() if 'num_batches_tracked' not in k}
    effnet.load_state_dict(new_state_dict)
    print("effnet loaded, please work this time!")
except Exception as e:
    print(f"effnet still busted: {e}")

try:
    # cnn good, no changes
    cnn = leaf_cnn(num_classes).to(device)
    cnn.load_state_dict(torch.load('models/cnn/classifier_model2_best.pth', map_location=device, weights_only=True))
    print("cnn loaded, all good")
except Exception as e:
    print(f"cnn failed: {e}")