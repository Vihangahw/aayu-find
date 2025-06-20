import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# transforms for unet, just resize and to tensor
unet_transform = A.Compose([
    A.Resize(256, 256),
    ToTensorV2()
])

# transforms for resnet and efficientnet, imagenet norms
resnet_effnet_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# cnn transform, simple normalize
cnn_transform = A.Compose([
    A.Resize(256, 256),
    ToTensorV2()
])

# blur background using unet mask
def blur_background(img, mask, kernel=(21, 21)):
    blurred_img = cv2.GaussianBlur(img, kernel, 10)  # blur the whole image
    result = np.where(mask[..., np.newaxis] > 0.5, img, blurred_img)  # keep leaf, blur rest
    return result