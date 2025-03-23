# scripts/train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import yaml
from models.model import AnimalClassifier
from utils.data_utils import AnimalsDataset
from utils.transforms import get_transforms
from utils.logging import setup_logging, close_logging

def train():
    # Load configs
    with open('configs/dataset.yaml', 'r') as f:
        dataset_config = yaml.safe_load(f)
    with open('configs/hyperparameters.yaml', 'r') as f:  # Đổi tên file config
        hyperparam_config = yaml.safe_load(f)
    
    # Setup logging
    logger, writer = setup_logging()
    
    # Hyperparameters
    num_epochs = hyperparam_config['training']['num_epochs']
    batch_size = hyperparam_config['training']['batch_size']
    learning_rate = hyperparam_config['training']['learning_rate']
    optimizer_type = hyperparam_config['training']['optimizer'].lower()
    weight_decay = hyperparam_config['training']['weight_decay']
    momentum = hyperparam_config['training']['momentum']
    lr_scheduler_type = hyperparam_config['training']['lr_scheduler'].lower()
    step_size = hyperparam_config['training']['step_size']
    gamma = hyperparam_config['training']['gamma']
    backbone_name = hyperparam_config['training']['backbone']
    
    # Prepare datasets
    train_transforms = get_transforms('configs/transforms.yaml', phase='train')
    val_transforms = get_transforms('configs/transforms.yaml', phase='test')
    
    train_dataset = AnimalsDataset(
        root_dir=os.path.join(dataset_config['data']['processed_path'], 'train'),
        transform=train_transforms
    )
    val_dataset = AnimalsDataset(
        root_dir=os.path.join(dataset_config['data']['processed_path'], 'val'),
        transform=val_transforms
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = AnimalClassifier(num_classes=len(train_dataset.classes), backbone_name=backbone_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, 
                            weight_decay=weight_decay)
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_type} không được hỗ trợ!")
    
    # Learning rate scheduler
    if lr_scheduler_type == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler_type == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif lr_scheduler_type == 'none':
        scheduler = None
    else:
        raise ValueError(f"LR Scheduler {lr_scheduler_type} không được hỗ trợ!")
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Train phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        # Logging
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            logger.info(f"Saved best model with Val Acc: {best_val_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), 'checkpoints/final_model.pth')
    logger.info(f"Training completed. Best Val Acc: {best_val_acc:.2f}%")
    close_logging(writer)

if __name__ == "__main__":
    train()