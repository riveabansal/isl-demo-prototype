import torch
from torch import nn

class LandmarkMLP(nn.Module):
    """Small static two-hand landmark baseline, not HWGAT."""
    def __init__(self,classes):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Dropout(.15),
                                  nn.Linear(128,64),nn.ReLU(),nn.Dropout(.10),nn.Linear(64,classes))
    def forward(self,x):return self.layers(x)
