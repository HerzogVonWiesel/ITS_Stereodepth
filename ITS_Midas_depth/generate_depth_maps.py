import shutil
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
import os

def clear_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    
def generate_depth_maps(input_folder, output_folder, model_path, model_type):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else :
        clear_folder(output_folder)

        # MiDaS
    #model_path = "midas_model.pt"
    #model_type = "MiDaS_small" 
    model_path = "midas_large_model.pt"
    model_type = "DPT_Large"  
    model = torch.hub.load("intel-isl/MiDaS", model_type)

    # Charge models weight
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # CPU
    device = torch.device("cpu")
    model.to(device)

    transform = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)

            input_image = Image.open(image_path).convert("RGB")
            input_batch = transform(input_image).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=input_image.size[::-1],
                mode="bicubic",
                align_corners=False
            ).squeeze()

            output_image = prediction.cpu().numpy()
            output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
            output_image = (output_image * 255).astype(np.uint8)

            Image.fromarray(output_image).save(output_image_path)
            print(f"Image de profondeur sauvegardée : {output_image_path}")

if __name__ == "__main__":
    input_folder = "frames"
    output_folder = "depth_maps"
    model_path = "midas_large_model.pt"
    model_type = "DPT_Large"
    generate_depth_maps(input_folder, output_folder, model_path, model_type)
