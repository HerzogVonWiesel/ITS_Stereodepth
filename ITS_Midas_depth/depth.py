from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np


image_path = "mountain.png"
output_image_path = "depth.png"

# MiDaS
#model_path = "midas_model.pt"
#model_type = "MiDaS_small" 
model_path = "midas_large_model.pt"
model_type = "DPT_Large"  
model = torch.hub.load("intel-isl/MiDaS", model_type)

# Charger les poids du modèle
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# CPU
device = torch.device("cpu")
model.to(device)

input_image = Image.open(image_path).convert("RGB")  # Conversion RGB
transform = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),                  # Conversion tensor PyTorch
    T.Normalize(mean=[0.5], std=[0.5])  # Normalisation for MiDaS
])

input_batch = transform(input_image).unsqueeze(0).to(device)

# No grad to save memory
with torch.no_grad():
    prediction = model(input_batch)

# Redimension
prediction = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=input_image.size[::-1],
    mode="bicubic",
    align_corners=False
).squeeze()

# Normalise bewteen 0 and 255
output_image = prediction.cpu().numpy()
output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())  # 0 et 1
output_image = (output_image * 255).astype(np.uint8)  # Convert into integer

Image.fromarray(output_image).save(output_image_path)
print(f"Image de profondeur sauvegardée : {output_image_path}")