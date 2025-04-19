import os

data_dir = '../data/processed'  # Adjust if your path is different
class_counts = {}

for class_folder in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_folder)
    if os.path.isdir(class_path):
        num_files = len([f for f in os.listdir(class_path) if f.endswith('.jpg')])
        class_counts[class_folder] = num_files

# Print sorted from largest to smallest
print("📊 Image count per class:")
for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{cls}: {count}")
