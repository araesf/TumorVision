import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import sys

from cnn_layering import TumorClassifier

# Define constants for mean and std values
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# 2. Load the Trained Model
def load_model(model_path, device):
    model = TumorClassifier(num_classes=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

# 3. Image Preprocessing
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)  # Add timeout
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the image: {e}")
        return None
    except (Image.UnidentifiedImageError, IOError) as e:
        print(f"Invalid image file: {e}")
        return None

def classify_mri_from_url(url, model, device):
    image = load_image_from_url(url)
    if image is None:
        print("Could not load the image. Please provide a valid image URL.")
        return None

    image_tensor = inference_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        pred_class_idx = outputs.argmax(dim=1).item()
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    predicted_class = classes[pred_class_idx]
    print(f"Predicted Class: {predicted_class}")

    # "No Tumor" if predicted_class == "notumor", else "Tumor Detected"
    return "No Tumor" if predicted_class == "notumor" else "Tumor Detected"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/Users/araesfarjani/TumorVision/best_model.pth"

    if len(sys.argv) > 1:
        image_url = sys.argv[1]
    else:
        image_url = input("Enter the URL of the MRI image: ")

    model = load_model(model_path, device)
    result = classify_mri_from_url(image_url, model, device)
    if result:
        print(result)
