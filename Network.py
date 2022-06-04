import torch
import torch.nn as nn
import torch.nn.functional as func


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(20531, 4096)
        self.l2 = nn.Linear(4096, 1024)
        self.l3 = nn.Linear(1024, 256)
        self.l4 = nn.Linear(256, 32)
        self.l5 = nn.Linear(32, 3)

    def forward(self, x: torch.Tensor):
        x = func.relu(self.l1(x))
        x = func.relu(self.l2(x))
        x = func.relu(self.l3(x))
        x = func.relu(self.l4(x))

        # with softmax
        # return func.softmax(self.l3(x), dim=-1)
        return self.l5(x)
