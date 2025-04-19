import matplotlib.pyplot as plt
import pandas as pd

# Load training log
log = pd.read_csv('../models/training_log.txt')

# Create subplots (2 rows, 1 column)
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# --- Plot 1: Loss ---
axs[0].plot(log['Epoch'], log['Train Loss'], label='Train Loss')
axs[0].plot(log['Epoch'], log['Validation Loss'], label='Val Loss')
axs[0].set_title('Training and Validation Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].set_xticks([9, 19, 29, 39, 49, 59, 69])
axs[0].grid()
axs[0].legend()

# --- Plot 2: Accuracy ---
axs[1].plot(log['Epoch'], log['Validation Accuracy'], label='Val Accuracy', color='green')
axs[1].set_title('Validation Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy (%)')
axs[1].set_xticks([9, 19, 29, 39, 49, 59, 69])
axs[1].grid()
axs[1].legend()

plt.tight_layout()
plt.show()
