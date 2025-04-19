import os
from PIL import Image

# Define input/output paths
RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"
TARGET_SIZE = (224, 224)

# Make sure processed dir exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Loop through images in raw dir
for filename in os.listdir(RAW_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            # Open and resize
            img_path = os.path.join(RAW_DIR, filename)
            img = Image.open(img_path).convert("RGB")
            img = img.resize(TARGET_SIZE)

            # Save to processed folder
            save_path = os.path.join(PROCESSED_DIR, filename)
            img.save(save_path)

            print(f"Processed: {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
