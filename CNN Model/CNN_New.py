import os
import librosa
import numpy as np
import torch
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================
# 1. Feature Extraction Function
# ============================================

def extract_features_cnn(file_path, sr=44100, n_mfcc=20, fixed_length=258):
    """
    Extracts audio features and ensures that each feature array has a fixed number of frames.

    Parameters:
        file_path (str): Path to the audio file.
        sr (int): Sampling rate.
        n_mfcc (int): Number of MFCCs to extract.
        fixed_length (int): Desired number of frames for consistency.

    Returns:
        np.ndarray: 2D array of stacked features with shape (25, fixed_length).
    """
    try:
        y, sr = librosa.load(file_path, sr=sr)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        y = np.zeros(sr)  # 1 second of silence or handle as needed

    # Extract features
    rms = librosa.feature.rms(y=y).flatten()
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr).flatten()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).flatten()
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Stack features into a 2D "image"
    features = np.vstack([rms, spec_cent, spec_bw, rolloff, zcr, mfcc])

    # Handle variable number of frames
    if features.shape[1] < fixed_length:
        pad_width = fixed_length - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :fixed_length]

    return features  # Shape: (25, fixed_length)

# ============================================
# 2. Dataset Class
# ============================================

class InstrumentDataset(Dataset):
    def __init__(self, dataset_path, instruments, label_encoder, transform=None, fixed_length=258):
        """
        Initializes the dataset by collecting file paths and corresponding labels.

        Parameters:
            dataset_path (str): Path to the dataset directory.
            instruments (list): List of instrument names.
            label_encoder (LabelEncoder): Fitted label encoder.
            transform (callable, optional): Optional transform to be applied on a sample.
            fixed_length (int): Fixed number of frames for feature consistency.
        """
        self.dataset_path = dataset_path
        self.instruments = instruments
        self.label_encoder = label_encoder
        self.transform = transform
        self.fixed_length = fixed_length

        # Collect file paths and labels
        self.file_paths = []
        self.labels = []

        for instrument in instruments:
            folder_path = os.path.join(dataset_path, instrument)
            if not os.path.isdir(folder_path):
                print(f"Warning: Folder {folder_path} does not exist.")
                continue
            for filename in os.listdir(folder_path):
                if filename.lower().endswith('.wav'):  # Ensure only WAV files are processed
                    full_path = os.path.join(folder_path, filename)
                    self.file_paths.append(full_path)
                    self.labels.append(instrument)

        # Encode labels (Assuming label_encoder is already fitted)
        self.labels = label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Retrieves the feature tensor and label for a given index.

        Parameters:
            idx (int): Index of the sample.

        Returns:
            tuple: (features, label)
                - features (torch.Tensor): Tensor of shape (1, 25, fixed_length).
                - label (int): Encoded label.
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Extract features
        features = extract_features_cnn(file_path, fixed_length=self.fixed_length)
        features = torch.tensor(features, dtype=torch.float32)

        # Normalize features (per sample normalization)
        features = (features - features.mean()) / (features.std() + 1e-6)

        # Apply transformations if provided
        if self.transform:
            features = self.transform(features)

        # Add a channel dimension for CNN
        features = features.unsqueeze(0)  # Shape: (1, 25, fixed_length)

        return features, label

# ============================================
# 3. Label Encoding and Dataset Preparation
# ============================================

# Dataset path and instruments
dataset_path = "./../IRMAS/IRMAS-TrainingData"  # Update this path as needed
instruments = 'flu pia tru org gac voi cel cla gel sax vio'.split()

# Initialize and fit the label encoder
label_encoder = LabelEncoder()

# Collect all labels to fit the encoder
all_labels = []
for instrument in instruments:
    folder_path = os.path.join(dataset_path, instrument)
    if not os.path.isdir(folder_path):
        print(f"Warning: Folder {folder_path} does not exist.")
        continue
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.wav'):
            all_labels.append(instrument)

label_encoder.fit(all_labels)

# Create dataset
fixed_length = 258  # Ensure consistency between feature extraction and model
total_dataset = InstrumentDataset(
    dataset_path=dataset_path,
    instruments=instruments,
    label_encoder=label_encoder,
    fixed_length=fixed_length
)

# Split dataset into training and test sets
dataset_size = len(total_dataset)
indices = list(range(dataset_size))
train_indices, test_indices = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=total_dataset.labels
)

# Subset datasets
train_subset = torch.utils.data.Subset(total_dataset, train_indices)
test_subset = torch.utils.data.Subset(total_dataset, test_indices)

# Save the label encoder for later use
with open("TITAN/label_encoder_cnn.pkl", "wb") as file:
    pickle.dump(label_encoder, file)

# ============================================
# 4. CNN Model Definition
# ============================================

class AudioCNN(nn.Module):
    def __init__(self, num_classes, fixed_length=258):
        """
        Initializes the CNN model for audio classification.

        Parameters:
            num_classes (int): Number of instrument classes.
            fixed_length (int): Fixed number of frames in input features.
        """
        super(AudioCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.dropout = nn.Dropout(0.5)

        # Calculate flattened size dynamically
        dummy_input = torch.zeros(1, 1, 25, fixed_length)  # (Batch size, Channels, Height, Width)
        with torch.no_grad():
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = self.pool(x)
            flattened_size = x.numel()

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the CNN.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 25, fixed_length).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ============================================
# 5. Model Initialization
# ============================================

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_classes = len(instruments)  # Number of instrument classes
model = AudioCNN(num_classes=num_classes, fixed_length=fixed_length).to(device)
print(model)

# ============================================
# 6. Loss Function and Optimizer
# ============================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjusted learning rate

# ============================================
# 7. Training Loop
# ============================================

# Hyperparameters
epochs = 50  # Increased number of epochs for better convergence
batch_size = 1024

# DataLoader for training
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize lists to store metrics
train_accuracies = []
epoch_losses = []

print("Starting training...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    epoch_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    # Optionally, you can add validation here if you create a validation set

# Save the trained model
torch.save(model.state_dict(), "cnn_instrument_recognizer.pth")
print("Model saved as cnn_instrument_recognizer.pth")

# ============================================
# 8. Plotting Training Metrics
# ============================================

# Plot training accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_accuracies, marker='o', label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy per Epoch')
plt.grid(True)
plt.legend()
plt.savefig("training_accuracy.png")
plt.show()

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), epoch_losses, marker='o', label='Training Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.grid(True)
plt.legend()
plt.savefig("training_loss.png")
plt.show()

# ============================================
# 9. Evaluation on Test Set
# ============================================

# DataLoader for testing
test_loader = DataLoader(test_subset, batch_size=128, shuffle=False, num_workers=4)

# Evaluate the model
model.eval()
all_preds = []
all_labels = []

print("Evaluating on test set...")

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision (Macro): {precision:.4f}")
print(f"Test Recall (Macro): {recall:.4f}")

# ============================================
# 10. Confusion Matrix Visualization
# ============================================

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# ============================================
# 11. Additional Recommendations
# ============================================

# Optional: Save training metrics for later analysis
with open("TITAN/training_metrics.pkl", "wb") as f:
    pickle.dump({
        "train_accuracies": train_accuracies,
        "epoch_losses": epoch_losses
    }, f)

# Optional: Implement Early Stopping or Learning Rate Scheduling
# Example: Reduce learning rate on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                 factor=0.5, patience=3, 
                                                 verbose=True)

# Modify the training loop to include scheduler step
# (Uncomment and integrate if needed)

# for epoch in range(epochs):
#     # Training steps...
#     # After calculating epoch_loss
#     scheduler.step(epoch_loss)