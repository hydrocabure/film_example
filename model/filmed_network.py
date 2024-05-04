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
        super().__init__() 

        self.in_channels = in_channels 
        self.out_channels = out_channels 

        assert emb_channels == out_channels, "embeddingのチャンネル数とoutのチャンネル数は一致している必要があります。" 

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
        z = z * gamma + beta # ここがFiLM Layer相当の処理をしている
        z = self.last_activ(z) 
        z = z + y 

        return z 


class Network(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 layer_cfg,

                 ):
        super().__init__() 

        self.in_channels = in_channels 
        self.out_channels = out_channels 

        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool(3, 2),
        )

        self.layers = nn.ModuleList([]) 
        
        for idx, (n_block, in_chan, out_chan, kernel, stride, padding, emb) in enumerate(layer_cfg):
            for _ in range(n_block):
                self.layers.append(ResnetBlock(in_chan, out_chan, kernel, stride, padding, emb))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Linear(out_channels, 100) 

    def forward(self, x):
        y = self.preprocess(x) 
        
        for layer in self.layers:
            y = layer(y)

        y = self.avgpool(y) 
        y = y.reshape(y.shape[0], -1) 
        y = self.fc(y) 

        return y 
    
