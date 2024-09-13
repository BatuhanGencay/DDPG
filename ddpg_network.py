import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device="cpu"):
        super(Critic, self).__init__()
        self.device = device
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state, action):

        x = torch.cat([state, action], 1).to(self.device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4, device="cpu"):
        super(Actor, self).__init__()
        self.device = device
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.head(x)

        steer = 0.6 * x[:, 0].reshape(-1, 1)
        accel_brake = x[:, 1].reshape(-1, 1)

        steer = torch.tanh(steer)
        accel_brake = torch.sigmoid(accel_brake).reshape(-1, 1)

        return torch.cat((steer, accel_brake), 1)

