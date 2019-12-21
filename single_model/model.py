import torch
from torch import nn, sigmoid, tanh, relu
import torch.nn.functional as F

torch.manual_seed(0)

# Loss function
@torch.jit.script
def loss_fn(pred, target):
    return F.nll_loss(input=pred, target=target)


class Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 15)
        self.fc2 = nn.Linear(15, 15)
        self.fc3 = nn.Linear(15, 15)
        self.fc4 = nn.Linear(15, 10)
        self.fc_out = nn.Linear(10, 3)

    def forward(self, x):

        x = sigmoid(self.fc1(x))
        x = sigmoid(self.fc2(x))
        x = sigmoid(self.fc3(x))
        x = sigmoid(self.fc4(x))
        x = F.log_softmax(self.fc_out(x), dim=1)
        return x
