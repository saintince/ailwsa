import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import *
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Resize((128,128)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))])
dataset = UTKFaceDataset(r'C:\Users\daa-a\PycharmProjects\ailwsa\data\UTKFace', transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = InfoNet().to(device)

criterion_age = nn.MSELoss()
criterion_gender = nn.MSELoss()
# criterion_height = nn.MSELoss()
# criterion_weight = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(len(loader)):
    model.train()
    total_loss = 0
    for images, age, gender in loader:
        images = images.to(device)
        age = age.unsqueeze(1).float().to(device)
        gender = gender.unsqueeze(1).float().to(device)
        # height = height.unsqueeze(1).float().to(device)
        # weight = weight.unsqueeze(1).float().to(device)

        outputs = model(images)

        loss_age = criterion_age(outputs['age'], age)
        loss_gender = criterion_gender(outputs['gender'], gender)
        # loss_height = criterion_height(outputs['height'], height)
        # loss_weight = criterion_weight(outputs['weight'], weight)

        loss = loss_age + loss_gender

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 100 == 0:
        print(f"epoch [{epoch}/{len(loader)}], Loss: {total_loss/len(loader)}")

torch.save(model.state_dict(), "model.pth")

















