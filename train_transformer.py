"""
Vision Transformer (ViT) for MNIST Digit Classification

This script demonstrates how to build and train a Vision Transformer from scratch.
Transformers work by splitting images into patches and using attention mechanisms
to let patches "communicate" with each other.

Key concepts:
- Patch embedding: Split image into small patches (tokens)
- Multi-head attention: Let patches attend to each other
- Positional embeddings: Encode spatial information
- Transformer blocks: Self-attention + feed-forward networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import os


# ============================================================================
# PATCH EMBEDDING
# ============================================================================
class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them."""
    
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        batch_size = x.shape[0]
        
        # Reshape into patches
        # (batch, 1, 28, 28) -> (batch, 49, 16) for patch_size=4
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, -1, self.patch_size * self.patch_size)
        
        # Embed patches
        x = self.proj(x)  # (batch, n_patches, embed_dim)
        
        return x


# ============================================================================
# MULTI-HEAD ATTENTION
# ============================================================================
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, n_patches, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch, n_patches, embed_dim * 3)
        qkv = qkv.reshape(batch_size, n_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, n_patches, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(batch_size, n_patches, embed_dim)
        out = self.out(out)
        
        return out


# ============================================================================
# TRANSFORMER BLOCK
# ============================================================================
class TransformerBlock(nn.Module):
    """A single transformer encoder block."""
    
    def __init__(self, embed_dim=64, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head attention
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        
        # Feed-forward network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


# ============================================================================
# VISION TRANSFORMER
# ============================================================================
class VisionTransformer(nn.Module):
    """Vision Transformer for MNIST."""
    
    def __init__(self, img_size=28, patch_size=4, in_channels=1, num_classes=10,
                 embed_dim=64, depth=4, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Class token (learnable token prepended to sequence)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, n_patches + 1, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Layer norm
        x = self.norm(x)
        
        # Classification (use class token)
        cls_output = x[:, 0]  # Take the class token
        out = self.head(cls_output)
        
        return out


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================
def main():
    print("=" * 70)
    print("Vision Transformer Training for MNIST")
    print("=" * 70)
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✓ Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"✓ Using CPU")
    
    print()
    
    # Data preparation
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print(f"✓ Training samples: {len(train_dataset):,}")
    print(f"✓ Test samples: {len(test_dataset):,}")
    print()
    
    # Model creation
    print("Creating Vision Transformer...")
    model = VisionTransformer(
        img_size=28,
        patch_size=4,
        in_channels=1,
        num_classes=10,
        embed_dim=64,
        depth=4,
        num_heads=4,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {total_params:,}")
    print(f"✓ Image patches: {model.patch_embed.n_patches} (4×4 each)")
    print(f"✓ Transformer depth: {len(model.blocks)} blocks")
    print(f"✓ Attention heads: {model.blocks[0].attn.num_heads}")
    print()
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    num_epochs = 10
    
    print(f"Training for {num_epochs} epochs...")
    print("=" * 70)
    
    start_time = time.time()
    best_test_acc = 0.0
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/vision_transformer_mnist.pth')
        
        # Print progress
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:5.2f}%")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print("=" * 70)
    print(f"\n✓ Training completed in {training_time:.2f} seconds")
    print(f"✓ Best test accuracy: {best_test_acc:.2f}%")
    print(f"✓ Throughput: {len(train_dataset) * num_epochs / training_time:.0f} images/second")
    print(f"✓ Model saved to: models/vision_transformer_mnist.pth")
    print()
    
    # Architecture summary
    print("=" * 70)
    print("Architecture Summary:")
    print("=" * 70)
    print(f"Input: 28×28 grayscale images")
    print(f"Patches: 49 patches of 4×4 pixels (16 values each)")
    print(f"Embedding: 64 dimensions per patch")
    print(f"Positional encoding: Learnable embeddings")
    print(f"Transformer blocks: 4 layers")
    print(f"  - Multi-head attention: 4 heads")
    print(f"  - Feed-forward: 4× expansion (64 → 256 → 64)")
    print(f"  - Residual connections + Layer normalization")
    print(f"Classification head: 64 → 10 (digit classes)")
    print(f"Total parameters: {total_params:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
