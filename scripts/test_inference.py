import os
import torch
from torchvision import transforms
from PIL import Image
from trained_model import PokemonClassifier

# --- Settings ---
model_path = '../models/pokemon_model.pth' #adjust if different path and name model
test_dir = '../data/test_inference_samples'
train_dir = '../data/training_data_augmentation'  # to get class names
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load class names based on training folder ---
class_names = sorted(os.listdir(train_dir))

# --- Load the model ---
model = PokemonClassifier(num_classes=len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.to(device)
model.eval()

# --- Image transform (same as training) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Run predictions ---
print("\n🧪 Running inference on test samples:\n")

for label in os.listdir(test_dir):
    class_folder = os.path.join(test_dir, label)
    if not os.path.isdir(class_folder):
        continue

    for image_name in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = output.argmax(dim=1).item()
            predicted_label = class_names[pred_idx]

        is_correct = predicted_label.lower() == label.lower()
        result_icon = "✅" if is_correct else "❌"

        print(f"📸 Image: {image_name}")
        print(f"   ✅ True Label: {label}")
        print(f"   {result_icon} Predicted:  {predicted_label} (Index: {pred_idx})")
        print("-" * 40)
