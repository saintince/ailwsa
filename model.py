import torch
import torch.nn.functional as f
import torch.nn as nn

class InfoNet(nn.Module):
    def __init__(self):
        super(InfoNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.fc1 = nn.Linear(64*16*16, 128)
        self.fc_age = nn.Linear(128, 1)
        self.fc_gender = nn.Linear(128, 1)
        # self.fc_height = nn.Linear(128,1)
        # self.fc_weight = nn.Linear(128, 1)


    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x, 2)
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2)
        x = f.relu(self.conv3(x))
        x = f.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        x = f.relu(self.fc1(x))

        age = self.fc_age(x)
        gender = torch.sigmoid(self.fc_gender(x))
        # height = self.fc_height(x)
        # weight = self.fc_weight(x)
        return {
            'age': age,
            'gender': gender
        }