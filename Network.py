import torch
import torch.nn as nn
import torch.nn.functional as func


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
<<<<<<< HEAD
        self.l1 = nn.Linear(20531, 2048)
        self.l2 = nn.Linear(2048, 512)
        self.l3 = nn.Linear(512, 128)
        self.l4 = nn.Linear(128, 32)
        self.l5 = nn.Linear(32, 3)
=======
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 3)
>>>>>>> main


    def forward(self, x: torch.Tensor):
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = func.relu(self.fc4(x))

        # with softmax
        # return func.softmax(self.l3(x), dim=-1)
        return self.fc5(x)
