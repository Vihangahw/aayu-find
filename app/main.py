from fastapi import FastAPI, File, UploadFile, HTTPException
from app.models import my_unet
from app.schemas import PredictionResponse
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import io
import os
import logging
import cv2
import numpy as np
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AAYU-FIND Leaf ID")

# Configuration
CONFIG = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'image_size': (256, 256),
    'num_classes': 4,
    'classes': ['HeenBovitiya', 'Karapincha', 'Kowakka', 'YakiNaran'],
    'model_paths': {
        'unet': 'models/unet/unet_best.pth',
        'ensemble': 'models/ensemble/ensembleUNET_model_best.pth'
    },
    'photo_dir': 'dataset/photos',
    'predictions_dir': 'predictions',
    'segment_transform': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ]),
    'classify_transform': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'min_mask_sum': 1000,
    'min_contour_area': 4000,
    'min_confidence': 0.7
}

# Create directories
os.makedirs(CONFIG['photo_dir'], exist_ok=True)
os.makedirs(CONFIG['predictions_dir'], exist_ok=True)

# Load models
try:
    unet = my_unet().to(CONFIG['device'])
    unet.load_state_dict(torch.load(CONFIG['model_paths']['unet'], map_location=CONFIG['device'], weights_only=True))
    unet.eval()
    logger.info("U-Net loaded successfully")
except Exception as e:
    logger.error(f"Failed to load U-Net: {str(e)}")
    raise RuntimeError(f"U-Net loading failed: {str(e)}")

try:
    def create_resnet(num_classes):
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model

    def create_efficientnet(num_classes):
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        return model

    def create_custom_cnn(num_classes):
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    class EnsembleModel(nn.Module):
        def __init__(self, num_classes):
            super(EnsembleModel, self).__init__()
            self.models = nn.ModuleList([
                create_resnet(num_classes),
                create_custom_cnn(num_classes),
                create_efficientnet(num_classes)
            ])
            for model in self.models:
                for param in model.parameters():
                    param.requires_grad = False
            self.fc = nn.Linear(num_classes, num_classes)

        def forward(self, x):
            outputs = torch.stack([model(x) for model in self.models], dim=0)
            avg_output = torch.mean(outputs, dim=0)
            return self.fc(avg_output)

    ensemble = EnsembleModel(CONFIG['num_classes']).to(CONFIG['device'])
    ensemble_state_dict = torch.load(CONFIG['model_paths']['ensemble'], map_location=CONFIG['device'], weights_only=True)
    ensemble.load_state_dict(ensemble_state_dict)
    ensemble.eval()
    logger.info("Ensemble model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Ensemble model: {str(e)}")
    raise RuntimeError(f"Ensemble model loading failed: {str(e)}")

def is_blank_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    variance = np.var(gray)
    logger.info(f"Image variance: {variance}")
    return variance < 10

def detect_leaf(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0
    max_contour_area = max([cv2.contourArea(c) for c in contours], default=0)
    logger.info(f"Max contour area: {max_contour_area}")
    return max_contour_area >= CONFIG['min_contour_area'], max_contour_area

def blur_background(image, mask, kernel_size=(15, 15)):
    blurred = cv2.GaussianBlur(image, kernel_size, sigmaX=10)
    result = np.where(mask[..., np.newaxis], image, blurred)
    return result

def save_prediction_to_json(data, filename, status):
    date_str = datetime.now().strftime('%Y-%m-%d')
    file_path = os.path.join(CONFIG['predictions_dir'], f"{date_str}.json")
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = {"timestamp": timestamp, "filename": filename, "status": status, **data}
    
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump([], f)
        logger.info(f"Created new prediction file: {file_path}")
    
    with open(file_path, 'r') as f:
        try:
            data_list = json.load(f)
        except json.JSONDecodeError:
            data_list = []
    
    data_list.append(entry)
    
    with open(file_path, 'w') as f:
        json.dump(data_list, f, indent=4)
    logger.info(f"{status} logged to {file_path}")

@app.get('/health')
async def health_check():
    logger.info("Health check accessed")
    return {'status': 'OK'}

@app.post('/predict', response_model=PredictionResponse)
@app.post('/api/predict', response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {file.filename}")
        
        # Resize image to 256x256
        img = cv2.resize(img, CONFIG['image_size'], interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        filename = file.filename
        logger.info(f"Predict endpoint accessed for file: {filename}")

        # Check for blank image
        if is_blank_image(img):
            logger.warning(f"Blank image detected: {filename}")
            save_prediction_to_json({"detail": "Blank or invalid image provided"}, filename, "error")
            raise HTTPException(status_code=400, detail="Blank or invalid image provided")

        # Segment
        img_tensor = CONFIG['segment_transform'](img).unsqueeze(0).to(CONFIG['device'])
        with torch.no_grad():
            mask = unet(img_tensor)
            mask_pred = (torch.sigmoid(mask) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
            mask_sum = mask_pred.sum()
            logger.info(f"Mask sum for {filename}: {mask_sum}")

        # Check for valid leaf segmentation
        if mask_sum < CONFIG['min_mask_sum']:
            logger.warning(f"No leaf detected: low mask sum ({mask_sum}) for {filename}")
            save_prediction_to_json({"detail": "No leaf detected: insufficient segmentation"}, filename, "error")
            raise HTTPException(status_code=400, detail="No leaf detected: insufficient segmentation")

        is_leaf, contour_area = detect_leaf(mask_pred)
        if not is_leaf:
            logger.warning(f"No leaf detected: invalid shape (contour area: {contour_area}) for {filename}")
            save_prediction_to_json({"detail": "No leaf detected: invalid shape"}, filename, "error")
            raise HTTPException(status_code=400, detail="No leaf detected: invalid shape")

        # Blur background
        img_blurred = blur_background(img, mask_pred)
        segmented_img = Image.fromarray(img_blurred)

        # Save to photos
        save_path = os.path.join(CONFIG['photo_dir'], f"seg_{filename}")
        segmented_img.save(save_path)
        logger.info(f"Saved segmented image: {save_path}")

        # Classify
        segmented_tensor = CONFIG['classify_transform'](segmented_img).unsqueeze(0).to(CONFIG['device'])
        with torch.no_grad():
            ensemble_pred = ensemble(segmented_tensor).softmax(dim=1)
            logger.info(f"Ensemble pred: {ensemble_pred.tolist()}")
            pred_class_idx = ensemble_pred.argmax(dim=1).item()
            confidence = ensemble_pred[0, pred_class_idx].item()

        # Determine final class based on confidence threshold
        final_class = "CouldNotPredict" if confidence < CONFIG['min_confidence'] else CONFIG['classes'][pred_class_idx]

        # Create prediction response
        prediction = PredictionResponse(
            final_class=final_class,
            final_confidence=confidence if final_class != "CouldNotPredict" else 0.0
        )

        # Save prediction to JSON file
        save_prediction_to_json({
            "final_class": final_class,
            "final_confidence": confidence if final_class != "CouldNotPredict" else 0.0
        }, filename, "success")

        return prediction
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))