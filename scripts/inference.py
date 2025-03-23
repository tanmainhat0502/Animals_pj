# scripts/inference.py
import os
import torch
from PIL import Image
import argparse
from utils.transforms import get_transforms
from any_models.model import AnimalClassifier
import yaml
from utils.data_utils import AnimalsDataset
def inference(image_path):
    # Load config
    with open('configs/dataset.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test transforms
    transform = get_transforms('configs/transforms.yaml', phase='test')
    
    # Load model
    processed_path = config['data']['processed_path']
    train_dataset = AnimalsDataset(os.path.join(processed_path, 'train'))
    model = AnimalClassifier(num_classes=len(train_dataset.classes))
    model.load_state_dict(torch.load('models/final_model.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = train_dataset.classes[predicted.item()]
    
    print(f'Predicted class: {predicted_class}')
    return predicted_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Animal Classification Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()
    
    inference(args.image)