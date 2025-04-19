import torch
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from pytorch_nndct.apis import torch_quantizer

# Define your model class again
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 481)  

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the model
model = SimpleCNN()
model.load_state_dict(torch.load("pokemon_model.pth", map_location='cpu'))
model.eval()

# Dummy input for quantization
dummy_input = torch.randn(1, 1, 28, 28)

# Apply quantization
quantizer = torch_quantizer("calib", model, (dummy_input,), output_dir="quantized_model_output")
quantized_model = quantizer.quant_model

# Save the quantized model for deployment
torch.save(quantized_model.state_dict(), "model_quantized.pth")
