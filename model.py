import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import math

class LightweightViT(nn.Module):
    """Lightweight Vision Transformer for student model"""
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=768, depth=6, num_heads=12, 
                 mlp_ratio=4., num_classes=3):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # DINOv2 uses 14x14 patches for 224x224 images, resulting in 256 patches + 1 cls token = 257
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.1,
                batch_first=True
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def enable_mixed_precision(self):
        """Ensure model parameters are compatible with mixed precision training"""
        # Convert certain parameters to ensure compatibility
        for module in self.modules():
            if isinstance(module, (nn.LayerNorm, nn.Linear)):
                # Ensure biases are initialized properly
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data = module.bias.data.float()
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.data = module.weight.data.float()
            
    def forward_features(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # B, C, H, W -> B, embed_dim, H//patch_size, W//patch_size
        x = x.flatten(2).transpose(1, 2)  # B, embed_dim, N -> B, N, embed_dim
        
        # Add cls token and positional embedding
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return x  # Return all tokens including cls token
    
    def forward(self, x):
        x = self.forward_features(x)
        # Use cls token for classification
        cls_token = x[:, 0]
        return self.head(cls_token), x[:, 1:]  # Return classification and patch tokens


# class LSTMClassifier(nn.Module):
#     """LSTM module for temporal analysis of patch features"""
#     def __init__(self, input_dim=384, hidden_dim=256, num_layers=2, num_classes=3, dropout=0.5):
#         super().__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
#                            batch_first=True, dropout=dropout, bidirectional=True)
#         self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, batch_first=True)
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, num_classes)
#         )
#
#     def forward(self, x):
#         # x shape: (batch_size, seq_len, input_dim)
#         lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim*2)
#
#         # Apply attention to focus on important features
#         attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
#
#         # Global average pooling
#         pooled = torch.mean(attn_out, dim=1)  # (batch_size, hidden_dim*2)
#
#         # Classification
#         output = self.classifier(pooled)
#         return output

class LSTMClassifier(nn.Module):
    def __init__(self, CNN_embed_dim=768, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=50):
        super(LSTMClassifier, self).__init__()
        self.LSTM = nn.LSTM(input_size=CNN_embed_dim,
                            hidden_size=h_RNN,
                            num_layers=h_RNN_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(h_RNN, h_FC_dim)
        self.fc2 = nn.Linear(h_FC_dim, num_classes)
        self.drop_p = drop_p

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, _ = self.LSTM(x_RNN, None)
        x = self.fc1(RNN_out[:, -1, :])
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x


class FeatureAdapter(nn.Module):
    """Adapter to map student features to teacher feature space"""
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.adapter = nn.Linear(student_dim, teacher_dim)
        
    def forward(self, x):
        return self.adapter(x)


class StudentModel(nn.Module):
    """Complete student model combining lightweight ViT and LSTM for sequence processing"""
    def __init__(self, num_classes=3, sequence_length=60, embed_dim=768):
        super().__init__()
        self.vit = LightweightViT(num_classes=num_classes, embed_dim=embed_dim)
        self.lstm = LSTMClassifier(CNN_embed_dim=embed_dim, num_classes=num_classes)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        
    def forward(self, x):
        """
        Forward pass for sequence of images
        Args:
            x: Tensor of shape (batch_size, sequence_length, channels, height, width)
        Returns:
            combined_output: Classification logits
            sequence_features: Features for each frame in the sequence
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape to process all frames at once: (batch_size * seq_len, channels, height, width)
        x_flat = x.view(batch_size * seq_len, channels, height, width)
        
        # Get ViT features for all frames
        cls_outputs, patch_features = self.vit(x_flat)
        
        # Reshape back to sequence format
        # cls_outputs: (batch_size * seq_len, num_classes) -> (batch_size, seq_len, num_classes)
        cls_outputs = cls_outputs.view(batch_size, seq_len, -1)
        
        # patch_features: (batch_size * seq_len, num_patches, feature_dim) -> (batch_size, seq_len, num_patches, feature_dim)
        num_patches, feature_dim = patch_features.shape[1], patch_features.shape[2]
        patch_features = patch_features.view(batch_size, seq_len, num_patches, feature_dim)
        
        # For LSTM, we'll use the average of patch features for each frame
        # This gives us a sequence of frame-level features
        frame_features = torch.mean(patch_features, dim=2)  # (batch_size, seq_len, feature_dim)
        
        # Process frame sequence with LSTM
        lstm_output = self.lstm(frame_features)
        
        # Average the ViT classification outputs across the sequence
        avg_cls_output = torch.mean(cls_outputs, dim=1)
        
        # Combine ViT and LSTM outputs
        combined_output = avg_cls_output + lstm_output
        
        # Return features in the format expected by the loss function
        # We'll return the mean patch features across the sequence for distillation
        sequence_features = torch.mean(patch_features, dim=1)  # (batch_size, num_patches, feature_dim)
        
        return combined_output, sequence_features
    
    def enable_mixed_precision(self):
        """Ensure model parameters are compatible with mixed precision training"""
        self.vit.enable_mixed_precision()
        # Also apply to LSTM components
        for module in self.lstm.modules():
            if isinstance(module, (nn.LayerNorm, nn.Linear, nn.LSTM)):
                for param_name, param in module.named_parameters():
                    if param is not None:
                        param.data = param.data.float()


def get_dinov2_model(model_name='dinov2_vitb14'):
    """Load pre-trained DINOv2 model"""
    try:
        import torch.hub
        model = torch.hub.load('facebookresearch/dinov2', model_name)
        return model
    except Exception as e:
        print(f"Failed to load DINOv2 model: {e}")
        return None