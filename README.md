# PokéScan-AI  
AI model to recognize Pokémon trading cards using Kria board

## 📌 Project Overview  
PokéScan-AI is a small edge AI project built to detect and classify Pokémon trading cards. The goal is to run this on a Xilinx Kria board so it can recognize cards in real time.

Right now, the project can:
- Preprocess card images
- Work with a small Pokémon card image dataset
- Prepare images in a format ready for training

Planned (but not all completed):
- Deploy it using Vitis AI tools
- Connect Kria’s camera for real-time scanning

---

## 📁 Directory Structure  
- `data/` – raw and processed image folders  
- `scripts/` – Python scripts for preprocessing and training  
- `models/` – saved trained model  
- `vitis_ai/` – files related to compiling with Vitis AI  
- `requirements.txt` – Python dependencies  

---

## ⚙️ Setup  

### 1. **Install dependencies**  
Make sure Python is installed, then run:

pip install -r requirements.txt

### 2. **Preprocess images**  
To resize and format the dataset:

python scripts/preprocess.py

Additional scripts are provided to help organize and clean the dataset, including splitting it into training, validation, and testing sets, balancing class distribution, and filtering out underrepresented classes.

### 3. **Train the model**  
Once data is processed, you can train the model with:

python scripts/train_model.py

### 4. **Run inference**
Once the model is finish training, you can load the model with:

python scripts/test_inference.py

---

## ⚠️ Known Issues  
- Couldn’t get the model to accelerate using Vitis AI  
- Sensor/camera input with Kria didn’t work in time  
- Dataset was highly imbalanced (some classes had only 1 image)  
- Accuracy stayed low   
- Inference only works on local images for now (not live scanning)

---

## 🔍 What I Tried to Fix  
- Tried augmentation to balance the dataset but had limited results  
- Restructured dataset to improve class balance  
- Switched models (tried CNN and other lightweight options)  
- Inference testing was done locally using test images

---

## 🙌 Credits  
Created by Ming Lei  
Oakland University 

## 📁 Dataset Note  
Due to GitHub file size limitations, the full Pokémon card image dataset used for training and testing could not be uploaded to this repository.

The original dataset was downloaded from Kaggle:  
🔗 [Pokémon Card Database - Kaggle](https://www.kaggle.com/datasets/stevenu/pokemon-card-database)

The dataset is provided as one large unorganized set of images. Before training the model, I preprocessed the images and split them into training, validation, and testing folders using a 60/20/20 ratio.  

![image](https://github.com/user-attachments/assets/9717f7ed-758d-49d3-98f1-2dcc9c92f4cf)
![image](https://github.com/user-attachments/assets/bddcff10-e23d-44ba-bfed-aeb79abee389)

Only a small sample of inference images is included here for reference. The full dataset was used locally during development.


