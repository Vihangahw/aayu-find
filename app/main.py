from fastapi import FastAPI, File, UploadFile, HTTPException
from app.models import my_unet, resnet_model, effnet_model, leaf_cnn
from app.schemas import PredictionResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import os
import logging
from collections import Counter
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AAYU-FIND Leaf ID")

# config
CONFIG = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'image_size': (256, 256),
    'num_classes': 4,
    'classes': ['HeenBovitiya', 'Karapincha', 'Kowakka', 'YakiNaran'],
    'model_paths': {
        'unet': 'models/unet/unet_best.pth',
        'resnet': 'models/resnet/best_resnet.pth',
        'effnet': 'models/efficientnet/best_efficientnet.pth',
        'cnn': 'models/cnn/classifier_model2_best.pth'
    },
    'photo_dir': 'dataset/photos',
    'segment_transform': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()  # normalizes to [0,1]
    ]),
    'classify_transform': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# create photo_dir
os.makedirs(CONFIG['photo_dir'], exist_ok=True)

# load models
try:
    unet = my_unet().to(CONFIG['device'])
    unet.load_state_dict(torch.load(CONFIG['model_paths']['unet'], map_location=CONFIG['device'], weights_only=True))
    unet.eval()
    logger.info("U-Net loaded successfully")
except Exception as e:
    logger.error(f"Failed to load U-Net: {str(e)}")
    raise RuntimeError(f"U-Net loading failed: {str(e)}")

try:
    resnet = resnet_model(num_classes=CONFIG['num_classes']).to(CONFIG['device'])
    resnet_state_dict = torch.load(CONFIG['model_paths']['resnet'], map_location=CONFIG['device'], weights_only=True)
    resnet.load_state_dict({f'resnet.{k}': v for k, v in resnet_state_dict.items()})
    resnet.eval()
    logger.info("ResNet loaded successfully")
except Exception as e:
    logger.error(f"Failed to load ResNet: {str(e)}")
    raise RuntimeError(f"ResNet loading failed: {str(e)}")

try:
    effnet = effnet_model(num_classes=CONFIG['num_classes']).to(CONFIG['device'])
    effnet_state_dict = torch.load(CONFIG['model_paths']['effnet'], map_location=CONFIG['device'], weights_only=True)
    effnet.load_state_dict({k.replace('model.', 'effnet.'): v for k, v in effnet_state_dict.items() if 'num_batches_tracked' not in k})
    effnet.eval()
    logger.info("EfficientNet loaded successfully")
except Exception as e:
    logger.error(f"Failed to load EfficientNet: {str(e)}")
    raise RuntimeError(f"EfficientNet loading failed: {str(e)}")

try:
    cnn = leaf_cnn(num_classes=CONFIG['num_classes']).to(CONFIG['device'])
    cnn.load_state_dict(torch.load(CONFIG['model_paths']['cnn'], map_location=CONFIG['device'], weights_only=True))
    cnn.eval()
    logger.info("CNN loaded successfully")
except Exception as e:
    logger.error(f"Failed to load CNN: {str(e)}")
    raise RuntimeError(f"CNN loading failed: {str(e)}")

def blur_background(image, mask, kernel_size=(15, 15)):
    blurred = cv2.GaussianBlur(image, kernel_size, sigmaX=10)
    result = np.where(mask[..., np.newaxis], image, blurred)
    return result

@app.get('/health')
async def health_check():
    logger.info("Health check accessed")
    return {'status': 'OK'}

@app.post('/predict')
@app.post('/api/predict')
async def predict(file: UploadFile = File(...)):
    try:
        # read image
        img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {file.filename}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        filename = file.filename
        logger.info(f"Predict endpoint accessed for file: {filename}")

        # segment
        img_tensor = CONFIG['segment_transform'](img).unsqueeze(0).to(CONFIG['device'])
        with torch.no_grad():
            mask = unet(img_tensor)
            mask_pred = (torch.sigmoid(mask) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
            mask_sum = mask_pred.sum()
            logger.info(f"Mask sum for {filename}: {mask_sum}")

        # blur background
        if mask_sum == 0:
            logger.warning(f"Empty mask for {filename}, using original image")
            img_blurred = img
        else:
            img_blurred = blur_background(img, mask_pred)
        segmented_img = Image.fromarray(img_blurred)

        # save to photos
        save_path = os.path.join(CONFIG['photo_dir'], f"seg_{filename}")
        segmented_img.save(save_path)
        logger.info(f"Saved segmented image: {save_path}")

        # classify
        segmented_tensor = CONFIG['classify_transform'](segmented_img).unsqueeze(0).to(CONFIG['device'])
        with torch.no_grad():
            resnet_pred = resnet(segmented_tensor).softmax(dim=1)
            effnet_pred = effnet(segmented_tensor).softmax(dim=1)
            cnn_pred = cnn(segmented_tensor).softmax(dim=1)
            logger.info(f"ResNet pred: {resnet_pred.tolist()}")
            logger.info(f"EffNet pred: {effnet_pred.tolist()}")
            logger.info(f"CNN pred: {cnn_pred.tolist()}")

            # 2/3 voting
            resnet_class = resnet_pred.argmax(dim=1).item()
            effnet_class = effnet_pred.argmax(dim=1).item()
            cnn_class = cnn_pred.argmax(dim=1).item()
            votes = [resnet_class, effnet_class, cnn_class]
            vote_counts = Counter(votes)
            logger.info(f"Votes: {votes}, Counts: {vote_counts}")

            if vote_counts.most_common(1)[0][1] >= 2:  # majority
                pred_class = vote_counts.most_common(1)[0][0]
                confidences = [resnet_pred[0, pred_class].item(), effnet_pred[0, pred_class].item(), cnn_pred[0, pred_class].item()]
                confidence = sum(confidences) / len(confidences)
            else:  # no majority, pick highest confidence
                preds = [resnet_pred, effnet_pred, cnn_pred]
                confidences = [p.max().item() for p in preds]
                pred_class = preds[confidences.index(max(confidences))].argmax(dim=1).item()
                confidence = max(confidences)

        return PredictionResponse(
            resnet_class=CONFIG['classes'][resnet_class],
            resnet_confidence=resnet_pred[0, resnet_class].item(),
            effnet_class=CONFIG['classes'][effnet_class],
            effnet_confidence=effnet_pred[0, effnet_class].item(),
            cnn_class=CONFIG['classes'][cnn_class],
            cnn_confidence=cnn_pred[0, cnn_class].item(),
            final_class=CONFIG['classes'][pred_class],
            final_confidence=confidence
        )
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))