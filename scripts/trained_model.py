# scripts/train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# Placeholder transformation and dataset setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Replace with actual processed dataset directory
train_data_dir = "../data/processed"  # This is where your preprocessed images will go
train_dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# Use a simple pretrained model as a placeholder (e.g. ResNet18)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))  # Adjust for num of classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Placeholder training loop
print("Starting training...")
for epoch in range(1):  # Just 1 epoch for demo
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}] - Loss: {running_loss:.4f}")

# Save model (placeholder path)
torch.save(model.state_dict(), "../models/model_placeholder.pth")
print("Model saved to models/model_placeholder.pth")
