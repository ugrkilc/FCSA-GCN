import torch
import torch.nn as nn
from stgc import STGC

class BaseGFE(nn.Module):
    def __init__(self, layers, num_person, num_classes, out_channels):
        super().__init__()
        self.num_person = num_person
        self.layers = nn.ModuleList(layers)

    def forward(self, x, A):
        for layer in self.layers:
            x = layer(x, A)
        return x

class GFE_one(BaseGFE):
    def __init__(self, num_person, num_classes, dropout, residual, A_size, input_channels):
        shared_layer = STGC(64, 64, 1, dropout, residual, A_size)
        layers = [
            STGC(3, 64, 1, 0, False, A_size),
            shared_layer,
            shared_layer
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
    def __init__(self, num_person, num_classes, dropout, residual, A_25_size, A_11_size, A_6_size):
        super().__init__([], num_person, num_classes, 128)

        self.shared_layer_25 = STGC(128, 128, 1, dropout, residual, A_25_size)
        self.shared_layer_11 = STGC(128, 128, 1, dropout, residual, A_11_size)
        self.shared_layer_6 = STGC(128, 128, 1, dropout, residual, A_6_size)

        self.layer_25 = STGC(64, 128, 2, dropout, residual, A_25_size)
        self.layer_11 = STGC(64, 128, 2, dropout, residual, A_11_size)
        self.layer_6 = STGC(64, 128, 2, dropout, residual, A_6_size)

    def forward(self, x, A, mode):
        if mode == 25:
            x = self.layer_25(x, A)
            x = self.shared_layer_25(x, A)
            x = self.shared_layer_25(x, A)
        elif mode == 11:
            x = self.layer_11(x, A)
            x = self.shared_layer_11(x, A)
            x = self.shared_layer_11(x, A)
        elif mode == 6:
            x = self.layer_6(x, A)
            x = self.shared_layer_6(x, A)
            x = self.shared_layer_6(x, A)
        return x