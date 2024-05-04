import torch
from torch import nn 


class ResnetBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, 
                 stride,
                 padding, 
                 emb_channels 
                 ):
        
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.emb_channels  = emb_channels 

        self.convactiv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
        )

        self.convnorm = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels) 
        )

        self.last_activ = nn.ReLU() 

    def forward(self, x, emb):
        (beta, gamma) = emb 

        y = self.convactiv(x) 
        z = self.convnorm(y) 
        z = z * gamma + beta 
        z = self.last_activ(z) 
        z = z + y 

        return z 


class Network(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        
        self.in_channels = in_channels 
        self.out_channels = out_channels 

        self.layers = nn.ModuleList([]) 

