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
        for sequences, labels in dataloader:  # sequences now have shape (batch, seq_len, channels, height, width)
            sequences, labels = sequences.to(device), labels.to(device)
            outputs, _ = model(sequences)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def train_fold(fold_num, train_dir, val_dir, device, hyperparams):
    """Train a single fold"""
    print(f"\n=== Training Fold {fold_num} ===")
    
    # Extract hyperparameters
    batch_size = hyperparams['batch_size']
    num_epochs = hyperparams['num_epochs']
    learning_rate = hyperparams['learning_rate']
    validate_every = hyperparams['validate_every']
    sequence_length = hyperparams['sequence_length']
    patience = hyperparams['patience']
    
    # Create datasets and dataloaders
    train_loader, val_loader, class_to_idx = get_wave_dataloaders(
        train_dir, val_dir, batch_size=batch_size, sequence_length=sequence_length
    )
    
    print(f"Fold {fold_num} - Classes: {class_to_idx}")
    print(f"Fold {fold_num} - Training samples: {len(train_loader.dataset)}")
    print(f"Fold {fold_num} - Validation samples: {len(val_loader.dataset)}")
    
    # Initialize student model
    student_model = StudentModel(num_classes=len(class_to_idx), sequence_length=sequence_length, embed_dim=768).to(device)
    
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
        beta=1.0,     # Feature imitation loss weight
        gamma=1.0,    # Relational knowledge loss weight
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
    
    # Training history storage
    fold_train_losses = []
    fold_train_accs = []
    fold_val_losses = []
    fold_val_accs = []
    
    # Early stopping variables
    best_val_acc = 0.0
    epochs_without_improvement = 0
    best_model_path = f'best_student_model_fold_{fold_num}.pth'
    
    print(f"Starting training for fold {fold_num}...")
    
    for epoch in range(num_epochs):
        # Training phase
        student_model.train()
        running_loss = 0.0
        running_task_loss = 0.0
        running_feature_loss = 0.0
        running_relation_loss = 0.0
        train_preds = []
        train_labels = []
        
        epoch_start_time = time.time()
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast('cuda', enabled=torch.cuda.is_available()):
                # Student forward pass - x is now (batch_size, seq_len, channels, height, width)
                student_logits, student_features = student_model(sequences)
                
                # Teacher forward pass (no gradient)
                with torch.no_grad():
                    # Process each frame in the sequence through the teacher model
                    batch_size, seq_len = sequences.shape[:2]
                    # Reshape for teacher: (batch_size * seq_len, channels, height, width)
                    sequences_flat = sequences.view(batch_size * seq_len, *sequences.shape[2:])
                    
                    # DINOv2 returns intermediate layers
                    intermediate_layers = teacher_model.get_intermediate_layers(sequences_flat, n=1)
                    teacher_features_flat = intermediate_layers[0][:, 1:]  # Remove cls token
                    
                    # Reshape back to sequence and average across time
                    # teacher_features_flat: (batch_size * seq_len, num_patches, feature_dim)
                    num_patches, feature_dim = teacher_features_flat.shape[1], teacher_features_flat.shape[2]
                    teacher_features_seq = teacher_features_flat.view(batch_size, seq_len, num_patches, feature_dim)
                    # Average across sequence for distillation
                    teacher_features = torch.mean(teacher_features_seq, dim=1)  # (batch_size, num_patches, feature_dim)
                    
                    # Ensure teacher features are in the correct dtype for mixed precision
                    if torch.cuda.is_available() and sequences.device.type == 'cuda':
                        teacher_features = teacher_features.half()
                    
                    # Resize teacher features to match student features if needed
                    if teacher_features.shape[1] != student_features.shape[1]:
                        # Use interpolation to match dimensions
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
            
            # Training accuracy
            _, preds = torch.max(student_logits, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f'Fold {fold_num}, Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss_components["total_loss"].item():.4f}')
        
        # Calculate average losses and accuracy for the epoch
        avg_loss = running_loss / len(train_loader)
        avg_task_loss = running_task_loss / len(train_loader)
        avg_feature_loss = running_feature_loss / len(train_loader)
        avg_relation_loss = running_relation_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Store training metrics
        fold_train_losses.append(avg_loss)
        fold_train_accs.append(train_acc)
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        print(f'Fold {fold_num}, Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s')
        print(f'Training - Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}, Task Loss: {avg_task_loss:.4f}, '
              f'Feature Loss: {avg_feature_loss:.4f}, Relation Loss: {avg_relation_loss:.4f}')
        
        # Validation phase
        if (epoch + 1) % validate_every == 0:
            print("Validating model...")
            val_acc = validate_model(student_model, val_loader, device)
            fold_val_accs.append(val_acc)
            
            # For validation loss, we'll store 0 as a placeholder since it's expensive to compute
            fold_val_losses.append(0.0)
            
            print(f'Fold {fold_num}, Validation Accuracy: {val_acc:.4f}')
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                
                # Save best model
                torch.save({
                    'fold': fold_num,
                    'epoch': epoch,
                    'model_state_dict': student_model.state_dict(),
                    'adapter_state_dict': feature_adapter.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'train_losses': fold_train_losses,
                    'train_accs': fold_train_accs,
                    'val_losses': fold_val_losses,
                    'val_accs': fold_val_accs,
                }, best_model_path)
                print(f'New best model saved for fold {fold_num} with validation accuracy: {best_val_acc:.4f}')
            else:
                epochs_without_improvement += 1
                print(f'No improvement for {epochs_without_improvement} epochs')
                
                # Early stopping
                if epochs_without_improvement >= patience:
                    print(f'Early stopping triggered for fold {fold_num} after {epoch+1} epochs')
                    break
        else:
            # For non-validation epochs, append placeholders to keep arrays aligned
            fold_val_losses.append(0.0)
            fold_val_accs.append(0.0)
    
    print(f"Fold {fold_num} training completed!")
    print(f"Best validation accuracy for fold {fold_num}: {best_val_acc:.4f}")
    
    # Save final metrics for this fold
    fold_results_dir = f'fold_{fold_num}_results'
    os.makedirs(fold_results_dir, exist_ok=True)
    
    np.save(os.path.join(fold_results_dir, 'train_losses.npy'), np.array(fold_train_losses))
    np.save(os.path.join(fold_results_dir, 'train_accs.npy'), np.array(fold_train_accs))
    np.save(os.path.join(fold_results_dir, 'val_losses.npy'), np.array(fold_val_losses))
    np.save(os.path.join(fold_results_dir, 'val_accs.npy'), np.array(fold_val_accs))
    
    # Save final model
    final_model_path = f'final_student_model_fold_{fold_num}.pth'
    torch.save({
        'fold': fold_num,
        'model_state_dict': student_model.state_dict(),
        'adapter_state_dict': feature_adapter.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'train_losses': fold_train_losses,
        'train_accs': fold_train_accs,
        'val_losses': fold_val_losses,
        'val_accs': fold_val_accs,
    }, final_model_path)
    
    print(f"Final model for fold {fold_num} saved as '{final_model_path}'")
    print(f"Best model for fold {fold_num} saved as '{best_model_path}'")
    print(f"Metrics for fold {fold_num} saved in '{fold_results_dir}/'")
    
    return best_val_acc, fold_train_losses, fold_train_accs, fold_val_losses, fold_val_accs

def train_kfold():
    """Main training function with 5-fold cross validation"""
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    hyperparams = {
        'batch_size': 16,  # Reduced batch size due to sequence processing
        'num_epochs': 200,
        'learning_rate': 1e-4,
        'validate_every': 1,  # Validate every 5 epochs
        'sequence_length': 60,  # Number of images per sequence
        'patience': 100,  # Early stopping patience
    }
    
    print("Hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    # 5-fold cross validation
    num_folds = 5
    fold_results = []
    
    # Check if datasets_5fold directory exists
    datasets_root = Path('./datasets_5fold')
    if not datasets_root.exists():
        print(f"Warning: {datasets_root} not found. Using dummy fold structure for testing.")
        # Create dummy directories for testing
        os.makedirs(datasets_root, exist_ok=True)
        for fold in range(num_folds):
            fold_dir = datasets_root / f'fold_{fold}'
            os.makedirs(fold_dir / 'train', exist_ok=True)
            os.makedirs(fold_dir / 'val', exist_ok=True)
            print(f"Created dummy directory: {fold_dir}")
    
    overall_best_acc = 0.0
    overall_best_fold = -1
    
    for fold in range(num_folds):
        fold = 4
        
        # Define paths for this fold
        fold_dir = datasets_root / f'fold_{fold}'
        train_dir = fold_dir / 'train'
        val_dir = fold_dir / 'val'
        
        if not train_dir.exists() or not val_dir.exists():
            print(f"Skipping fold {fold}: directories {train_dir} or {val_dir} not found")
            continue
            
        try:
            # Train this fold
            fold_best_acc, train_losses, train_accs, val_losses, val_accs = train_fold(
                fold, train_dir, val_dir, device, hyperparams
            )
            
            fold_results.append({
                'fold': fold,
                'best_val_acc': fold_best_acc,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
            })
            
            # Track overall best
            if fold_best_acc > overall_best_acc:
                overall_best_acc = fold_best_acc
                overall_best_fold = fold
                
        except Exception as e:
            print(f"Error training fold {fold}: {e}")
            continue
    
    # Summary
    print("\n" + "="*50)
    print("5-FOLD CROSS VALIDATION RESULTS")
    print("="*50)
    
    if fold_results:
        fold_accs = [result['best_val_acc'] for result in fold_results]
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        
        for result in fold_results:
            print(f"Fold {result['fold']}: Best Val Acc = {result['best_val_acc']:.4f}")
        
        print("-"*30)
        print(f"Mean Validation Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Overall Best Fold: {overall_best_fold} (Acc: {overall_best_acc:.4f})")
        
        # Save overall results
        overall_results = {
            'fold_results': fold_results,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'best_fold': overall_best_fold,
            'best_acc': overall_best_acc,
            'hyperparams': hyperparams,
        }
        
        np.save('kfold_results.npy', overall_results)
        print(f"Overall results saved to 'kfold_results.npy'")
    else:
        print("No folds were successfully trained!")

if __name__ == "__main__":
    train_kfold()