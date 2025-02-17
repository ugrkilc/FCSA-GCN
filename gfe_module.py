import torch
import torch.nn as nn
import torch.nn.functional as F
from stgc import STGC 


class BaseGFE(nn.Module):

    def __init__(self, layers, num_person, num_classes, out_channels):
        super().__init__()
        self.num_person = num_person
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Conv2d(out_channels, num_classes, kernel_size=1, padding=0)

    def forward(self, x, A):
        for layer in self.layers:
            x = layer(x, A)
        return x

    def predict(self, x):
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0)//self.num_person, self.num_person, -1, 1, 1).mean(dim=1)
        x = self.fc(x)
        return x.view(x.size(0), -1)

class GFE_one(BaseGFE):

    def __init__(self, num_person, num_classes, dropout, residual, A_size, input_channels):
        layers = [
            STGC(3, 64, 1, 0, False, A_size),
            STGC(64, 64, 1, dropout, residual, A_size),
            STGC(64, 64, 1, dropout, residual, A_size)
        ]
        super().__init__(layers, num_person, num_classes, 64)          
      
        self.bn = nn.BatchNorm1d(input_channels * A_size[2] * num_person)

    def forward(self, x, A):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        x = super().forward(x, A)        
        return x

class GFE_two(BaseGFE):
    def __init__(self, num_person, num_classes, dropout, residual, A_size):
        layers = [
            STGC(64, 128, 2, dropout, residual, A_size),    
            STGC(128, 128, 1, dropout, residual, A_size),
            STGC(128, 128, 1, dropout, residual, A_size)
        ]
        super().__init__(layers, num_person, num_classes, 128)

    def forward(self, x, A):
        x = super().forward(x, A)
        # return x, self.predict(x)
        return x

