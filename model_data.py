import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np

from cnn_layering import TumorClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#########################################
# 1. Data & Transforms
#########################################
# ImageNet-like normalization
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# Where dataset is located (my root directory)
root_dir = "/Users/araesfarjani/Desktop/archive"

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_dataset = ImageFolder(os.path.join(root_dir, "Training"), transform=train_transforms)
val_dataset = ImageFolder(os.path.join(root_dir, "Testing"), transform=val_transforms)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Training samples:   {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Classes found: {train_dataset.classes}")

#########################################
# 2. Model Initialization
#########################################
model = TumorClassifier(num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#########################################
# 3. Training Loop
#########################################
num_epochs = 20
best_val_accuracy = 0.0

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    ########################
    # TRAIN
    ########################
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    ########################
    # VALIDATE
    ########################
    model.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)

            val_running_loss += val_loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss_avg = val_running_loss / len(val_loader)
    val_accuracy = correct / total

    val_losses.append(val_loss_avg)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {running_loss:.4f}, Train Acc: {train_accuracy:.2%}, "
          f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.2%}")
    
    checkpoint = {
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'best_val_accuracy': best_val_accuracy
    }
    torch.save(checkpoint, 'best_model.pth')

    # Save the best model checkpoint
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"** Saved new best model with val_acc={val_accuracy:.2%}")

print("Training complete.")
print(f"Best validation accuracy: {best_val_accuracy:.2%}")

#########################################
# 4. Visualize Training
#########################################
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss History')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot([acc*100 for acc in train_accuracies], label='Train Accuracy (%)')
plt.plot([acc*100 for acc in val_accuracies], label='Val Accuracy (%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy History')
plt.legend()

plt.tight_layout()
plt.show()
