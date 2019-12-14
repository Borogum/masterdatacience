import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(0)


class Classifier(nn.Module):

    def __init__(self):
        torch.manual_seed(0)  # Reproducibility
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 3)

    def forward(self, x):
        # make sure input tensor is flattened
        # x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
