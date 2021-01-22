import torch.nn as nn
import torch.nn.functional as F


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        print("Initializing MyNet...")
        self.l1 = nn.Linear(28*28, 128, bias=False)
        self.l2 = nn.Linear(128, 10, bias=False)

    def forward(self, x):
        if x.shape[1] != 28*28:
            raise Exception(f"Wrong size input for x : {x.shape[1]}, expected 28*28")
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
