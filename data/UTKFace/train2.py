import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import os
import shutil

model = InfoNet2()
model.load_state_dict(torch.load("model.pth"))
model.to("cuda")

# Добавляем новые выходы для роста и веса
model.fc_height = nn.Linear(128, 1)
model.fc_weight = nn.Linear(128, 1)

folders = [
    ("/content/bodym/testA/mask", "/content/bodym/testA/hwg_metadata.csv"),
    ("/content/bodym/testB/mask", "/content/bodym/testB/hwg_metadata.csv"),
    ("/content/bodym/train/mask", "/content/bodym/train/hwg_metadata.csv")
]

merged_rows = []
os.makedirs("/content/data_all", exist_ok=True)

for img_folder, csv_path in folders:
    df = pd.read_csv(csv_path, sep='\t')
    for i, row in df.iterrows():
        src = os.path.join(img_folder, row["image_id"])
        dst = os.path.join("/content/data_all", row["image_id"])
        shutil.copy(src, dst)
        merged_rows.append(row)

merged_df = pd.DataFrame(merged_rows)
merged_df.to_csv("/content/data_all/merged.csv", sep='\t', index=False)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = BodyMDataset("/content/data_all", "/content/data_all/merged.csv", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

criterion_h = nn.MSELoss()
criterion_w = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    model.train()
    total_loss = 0
    for images, gender, height, weight in loader:
        images = images.to("cuda")
        height = height.unsqueeze(1).float().to("cuda")
        weight = weight.unsqueeze(1).float().to("cuda")

        outputs = model(images)
        loss_h = criterion_h(outputs["height"], height)
        loss_w = criterion_w(outputs["weight"], weight)
        loss = loss_h + loss_w

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/30], Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "model_full.pth")
print("✅ Модель для возраста, пола, роста и веса сохранена")
