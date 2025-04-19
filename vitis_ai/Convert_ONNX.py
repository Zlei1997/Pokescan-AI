import torch
from your_model_file import YourModelClass  # Replace with actual model file and class

# Create model instance
model = YourModelClass()  # Replace with your model name
model.load_state_dict(torch.load("pokemon_model.pth"))
model.eval()

# input should match the image size used in your dataset (e.g., 224x224)
input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(model, input, "pokemon_model.onnx", 
                  input_names=["input"], output_names=["output"],
                  opset_version=11)

print("Export complete! pokemon_model.onnx created.")
