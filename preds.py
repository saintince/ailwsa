import torch
from torchvision import transforms
from PIL import Image
from model import *
from train import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'model.pth'
image_path = r"C:\Users\daa-a\PycharmProjects\ailwsa\data\UTKFace\27_0_0_20170116204122640.jpg.chip.jpg"


model = InfoNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)


with torch.no_grad():
    output = model(input_tensor)

pred_age = outputs['age'].item()
pred_gender = "Мужчина" if outputs['gender'].item() < 0.5 else "Женщина"

print("=== Результат ===")
print(f"Возраст (примерно): {pred_age:.1f} лет")
print(f"Пол: {pred_gender}")



































