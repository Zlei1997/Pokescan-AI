import os
import shutil
import random

# Paths
source_dir = '../data/PokemonCard_Testing' # adjust if different name
train_dir = '../data/Training' # adjust if different name
val_dir = '../data/Testing' # adjust if different name

# Ratio
train_ratio = 0.6  # 60%

# Create target directories if they don't exist
for base_dir in [train_dir, val_dir]:
    os.makedirs(base_dir, exist_ok=True)

# Go through each class folder
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue  # skip any files

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Create class subfolders in train/ and val/
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Move training images
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_dir, class_name, img)
        shutil.copy2(src, dst)

    # Move validation images
    for img in val_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_dir, class_name, img)
        shutil.copy2(src, dst)

    print(f"Split {class_name}: {len(train_images)} train, {len(val_images)} val")

print("✅ Dataset split complete!")
