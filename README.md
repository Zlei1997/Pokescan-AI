# Pokescan-AI
AI model to recognize Pokémon trading cards using Kria board

## 📌 Project Overview

PokéScan-AI is an edge AI project designed to detect and classify Pokémon trading cards using computer vision. The system preprocesses card images, prepares them for training, and will later deploy an optimized model to the Xilinx Kria board for real-time inference.

The goal is to recognize different Pokémon cards based on visual features like character artwork, text, and layout.

Currently, the project supports:
- Preprocessing card images (resize and convert to RGB)
- A small set of sample Pokémon card images
- Ready-to-use training data format

Planned next steps:
- Train a lightweight CNN classification model
- Deploy the model using Vitis AI tools on Kria
- Integrate live camera scanning from Kria board

## Setup

To get started with PokéScan-AI:

1. **Install dependencies**  
   Make sure you have Python installed. Then run:
pip install -r requirements.txt


2. **Preprocess images**  
This script resizes card images and prepares them for model training:

python scripts/preprocess.py


Processed images will be saved in the `data/processed/` directory (excluded from Git).

