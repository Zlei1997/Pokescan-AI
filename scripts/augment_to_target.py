import os
import random
from PIL import Image
from torchvision import transforms

# Folder containing all class folders
processed_dir = r"C:\Users\zlei9\OneDrive\Desktop\Pokescan-AI\data\training_data_augmentation" #depending on what directory and file name

# Augmentation transformations
augmentation = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
])

def augment_folder(folder_path, target=50, skip_class="Pikachu"):
    for class_name in os.listdir(folder_path):
        class_folder = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_folder):
            continue

        images = [f for f in os.listdir(class_folder) if f.endswith((".jpg", ".jpeg", ".png"))]
        current_count = len(images)

        # Skip Pikachu (or whatever skip_class is)
        if class_name == skip_class:
            print(f"Skipping {class_name} with {current_count} images.")
            continue

        if current_count >= target:
            print(f"{class_name} already has {current_count} images. Skipping.")
            continue

        print(f"Augmenting {class_name}: {current_count} → {target}")

        while len(os.listdir(class_folder)) < target:
            source_image = random.choice(images)
            source_path = os.path.join(class_folder, source_image)

            with Image.open(source_path).convert("RGB") as img:
                new_img = augmentation(img)
                new_name = f"aug_{random.randint(100000, 999999)}.jpg"
                new_img.save(os.path.join(class_folder, new_name))

augment_folder(processed_dir, target=50, skip_class="Pikachu")
