import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import csv
import os

class PokemonClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PokemonClassifier, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Updated: Use proper 60/20/20 split directory
    base_dir = "../data"

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(os.path.join(base_dir, "training_data_augmentation"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(base_dir, "validation_data"), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(base_dir, "testing_data"), transform=val_test_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Model setup
    model = PokemonClassifier(num_classes=len(train_dataset.classes))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Logging setup
    log_file = "training_log.txt"
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Validation Accuracy', 'Time(s)'])

    # Training loop
    epochs = 70
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation
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

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        epoch_time = round(time.time() - start_time, 2)

        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Accuracy={accuracy:.2f}%, Time={epoch_time}s")

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, round(avg_train_loss, 4), round(avg_val_loss, 4), round(accuracy, 2), epoch_time])

        if epoch % 10 == 0:
            checkpoint_path = f"model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)

    # Final evaluation (optional)
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")