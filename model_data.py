import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from cnn_layering import TumorNeuralNetwork  # Import the model
from data_processing import load_data  # Import the data loader
import torch.nn as nn

def train_model(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs.squeeze(), labels)
        
        # Backward pass and optimization
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += loss.item()
    
    # Return the average loss for the epoch
    return running_loss / len(train_loader)

def validate_model(val_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs.squeeze(), labels)
            running_loss += loss.item()
    
    return running_loss / len(val_loader)

def save_checkpoint(model, epoch, best_val_loss, model_save_path):
    model_save_path = f"{model_save_path}_epoch_{epoch+1}_val_loss_{best_val_loss:.4f}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model checkpoint saved at: {model_save_path}")

# Early stopping function if improvement starts showing signs of overfitting
def early_stopping(val_loss, best_val_loss, early_stopping_counter, patience, improvement_threshold=0.005):
    if (best_val_loss - val_loss) > improvement_threshold:
        best_val_loss = val_loss
        early_stopping_counter = 0

        # Flag to indicate a significant improvement
        return False, early_stopping_counter, best_val_loss, True 
    else:
        early_stopping_counter += 1
    
    if early_stopping_counter >= patience:
        return True, early_stopping_counter, best_val_loss, False  # Early stop triggered
    
    return False, early_stopping_counter, best_val_loss, False  # No early stop, no significant improvement

if __name__ == "__main__":
    root_dir = r'/Users/araesfarjani/Desktop/archive'
    labels = ["glioma", "meningioma", "notumor", "pituitary"]

    # Load the train and validation datasets
    train_dataset, val_dataset = load_data(root_dir, labels)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, optimizer, and loss function
    model = TumorNeuralNetwork().to(device)
    
    # Use Adam optimizer with L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    # Binary Cross-Entropy Loss for binary classification
    criterion = nn.BCELoss()

    # Learning rate scheduler with ReduceLROnPlateau
    # Learning rate scheduler with more patience before reducing the LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=5)


    # Initialize early stopping parameters
    best_val_loss = float('inf')
    patience = 10
    early_stopping_counter = 0
    model_save_path = 'best_model'

    # Training loop
    n_epochs = 100
    for epoch in range(n_epochs):
        train_loss = train_model(train_loader, model, criterion, optimizer, device)
        val_loss = validate_model(val_loader, model, criterion, device)

        # Step the learning rate scheduler based on validation loss
        scheduler.step(val_loss)

        # Print the current learning rate and losses
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Learning Rate: {current_lr:.6f}")

        # Early stopping check
        stop, early_stopping_counter, best_val_loss, significant_improvement = early_stopping(
            val_loss, best_val_loss, early_stopping_counter, patience, improvement_threshold=0.005
        )

        # Save the best model only if there is a significant improvement
        if significant_improvement:
            save_checkpoint(model, epoch, best_val_loss, model_save_path)
        
        if stop:
            print(f"Early stopping triggered at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f}")
            break
