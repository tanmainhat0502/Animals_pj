# scripts/evaluate.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.data_utils import AnimalsDataset
from utils.transforms import get_transforms
from models.model import AnimalClassifier
import yaml

def evaluate_model():
    # Load config
    with open('configs/dataset.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Prepare test dataset
    test_transforms = get_transforms('configs/transforms.yaml', phase='test')
    test_dataset = AnimalsDataset(
        root_dir=os.path.join(config['data']['processed_path'], 'test'),
        transform=test_transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load model
    model = AnimalClassifier(num_classes=len(test_dataset.classes))
    model.load_state_dict(torch.load('models/final_model.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluation metrics
    total_loss = 0.0
    correct = 0
    total = 0
    losses = []
    accuracies = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            batch_loss = loss.item()
            batch_accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)
            losses.append(batch_loss)
            accuracies.append(batch_accuracy)
    
    # Tính trung bình
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    
    # Tạo và lưu biểu đồ
    os.makedirs('results', exist_ok=True)
    
    # Accuracy curve
    plt.figure()
    plt.plot(accuracies, label='Batch Accuracy')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve on Test Set')
    plt.legend()
    plt.savefig('results/accuracy_curve.png')
    plt.close()
    
    # Loss curve
    plt.figure()
    plt.plot(losses, label='Batch Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Loss Curve on Test Set')
    plt.legend()
    plt.savefig('results/loss_curve.png')
    plt.close()

if __name__ == "__main__":
    evaluate_model()