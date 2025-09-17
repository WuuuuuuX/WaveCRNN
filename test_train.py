import sys
sys.path.append('/Users/wuxi/PycharmProjects/waveClass')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast
import os
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from model import StudentModel, get_dinov2_model, FeatureAdapter
from loss import HeterogeneousKnowledgeDistillationLoss
from dataset import get_wave_dataloaders

def validate_model(model, dataloader, device):
    """Validate the model and return accuracy"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def train_model():
    """Main training function"""
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 4  # Small batch size for testing
    num_epochs = 7  # Just a few epochs for testing
    learning_rate = 1e-4
    validate_every = 2  # Validate every 2 epochs
    
    # Create datasets and dataloaders
    train_dir = Path('./datasets/train')
    val_dir = Path('./datasets/val')
    
    train_loader, val_loader, class_to_idx = get_wave_dataloaders(
        train_dir, val_dir, batch_size=batch_size
    )
    
    print(f"Classes: {class_to_idx}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize student model
    student_model = StudentModel(num_classes=len(class_to_idx), embed_dim=768).to(device)
    
    # Enable mixed precision compatibility
    student_model.enable_mixed_precision()
    
    # Load teacher model (DINOv2 Base)
    teacher_model = get_dinov2_model('dinov2_vitb14')
    if teacher_model is None:
        print("Failed to load teacher model. Exiting.")
        return
    
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # Set to evaluation mode
    for param in teacher_model.parameters():
        param.requires_grad = False  # Freeze teacher model
    
    # Initialize feature adapter (student dim to teacher dim)
    # DINOv2 base has 768 dim, our student has 768 dim, so identity adapter
    feature_adapter = FeatureAdapter(768, 768).to(device)
    
    # Ensure feature adapter is also mixed precision compatible
    for param in feature_adapter.parameters():
        if param is not None:
            param.data = param.data.float()
    
    # Loss function
    criterion = HeterogeneousKnowledgeDistillationLoss(
        alpha=1.0,    # Task loss weight
        beta=0.5,     # Feature imitation loss weight
        gamma=0.3,    # Relational knowledge loss weight
        temperature=4.0
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        list(student_model.parameters()) + list(feature_adapter.parameters()),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Mixed precision training scaler
    scaler = GradScaler('cuda') if torch.cuda.is_available() else GradScaler()
    
    # Training loop
    best_val_acc = 0.0
    training_history = {
        'train_loss': [],
        'val_acc': [],
        'task_loss': [],
        'feature_loss': [],
        'relation_loss': []
    }
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        student_model.train()
        running_loss = 0.0
        running_task_loss = 0.0
        running_feature_loss = 0.0
        running_relation_loss = 0.0
        
        epoch_start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast('cuda', enabled=torch.cuda.is_available()):
                # Student forward pass
                student_logits, student_features = student_model(images)
                
                # Teacher forward pass (no gradient)
                with torch.no_grad():
                    # DINOv2 returns a dict, we need the last hidden state
                    # For simplicity, we'll use the patch tokens (without cls token)
                    intermediate_layers = teacher_model.get_intermediate_layers(images, n=1)
                    teacher_features = intermediate_layers[0][:, 1:]  # Remove cls token
                    
                    # Ensure teacher features are in the correct dtype for mixed precision
                    if torch.cuda.is_available() and images.device.type == 'cuda':
                        teacher_features = teacher_features.half()
                    
                    # Resize teacher features to match student features if needed
                    if teacher_features.shape[1] != student_features.shape[1]:
                        # Use interpolation to match dimensions
                        # teacher_features: [B, N_t, D] -> [B, N_s, D]
                        teacher_features = F.interpolate(
                            teacher_features.permute(0, 2, 1), 
                            size=student_features.shape[1], 
                            mode='linear', 
                            align_corners=False
                        ).permute(0, 2, 1)
                
                # Compute loss
                total_loss, loss_components = criterion(
                    student_logits, teacher_features, student_features, 
                    labels, feature_adapter
                )
            
            # Backward pass
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Statistics
            running_loss += loss_components['total_loss'].item()
            running_task_loss += loss_components['task_loss'].item()
            running_feature_loss += loss_components['feature_loss'].item()
            running_relation_loss += loss_components['relation_loss'].item()
            
            # Limit batches for quick testing
            if batch_idx >= 2:  # Just process a few batches for testing
                break
                
            if batch_idx % 1 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss_components["total_loss"].item():.4f}')
        
        # Calculate average losses for the epoch
        avg_loss = running_loss / 3  # We're only processing 3 batches
        avg_task_loss = running_task_loss / 3
        avg_feature_loss = running_feature_loss / 3
        avg_relation_loss = running_relation_loss / 3
        
        # Store in history
        training_history['train_loss'].append(avg_loss)
        training_history['task_loss'].append(avg_task_loss)
        training_history['feature_loss'].append(avg_feature_loss)
        training_history['relation_loss'].append(avg_relation_loss)
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s')
        print(f'Average Loss: {avg_loss:.4f}, Task Loss: {avg_task_loss:.4f}, '
              f'Feature Loss: {avg_feature_loss:.4f}, Relation Loss: {avg_relation_loss:.4f}')
        
        # Validation phase
        if (epoch + 1) % validate_every == 0:
            print("Validating model...")
            val_acc = validate_model(student_model, val_loader, device)
            training_history['val_acc'].append(val_acc)
            
            print(f'Validation Accuracy: {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student_model.state_dict(),
                    'adapter_state_dict': feature_adapter.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                }, 'best_student_model.pth')
                print(f'New best model saved with validation accuracy: {best_val_acc:.4f}')
    
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Save final model
    torch.save({
        'model_state_dict': student_model.state_dict(),
        'adapter_state_dict': feature_adapter.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
    }, 'final_student_model.pth')
    
    print("Final model saved as 'final_student_model.pth'")
    print("Best model saved as 'best_student_model.pth'")

if __name__ == "__main__":
    train_model()