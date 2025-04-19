import os
import shutil
from collections import Counter

# Set threshold
THRESHOLD = 20 #can switch number depending on what threshold you want

# Path to your processed images
data_path = '../data/Processed' #adjust if different folder name

# Count images per class
class_counts = {}
for class_name in os.listdir(data_path):
    class_path = os.path.join(data_path, class_name)
    if os.path.isdir(class_path):
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_counts[class_name] = len(image_files)

# Identify underrepresented classes
to_remove = [cls for cls, count in class_counts.items() if count < THRESHOLD]

# Function to safely delete a folder
def safe_rmtree(path):
    try:
        shutil.rmtree(path)
        print(f"✅ Removed: {path}")
    except PermissionError:
        print(f"❌ Permission denied while deleting: {path}")
    except Exception as e:
        print(f"⚠️ Could not remove {path}: {e}")

# Delete folders
for cls in to_remove:
    class_path = os.path.join(data_path, cls)
    safe_rmtree(class_path)

print("\n🎯 Cleanup complete. Removed the following underrepresented classes:")
for cls in to_remove:
    print(f" - {cls}")
