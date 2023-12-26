import torch
from torch import nn as nn
import torch.nn.functional as F
import types, typing
from typing import Callable

## The first dude uses weight tying therefore a bit cheaper. 
class auto_encoder_bb(nn.Module):
    def __init__(self, 
                 dims:list[int], 
                 activation:Callable = nn.GELU("tanh"),
                 **kwargs)->None:
        super().__init__(**kwargs)
        
        self.dims = dims
        self.depth = len(dims)
        self.weights = nn.ParameterList([torch.empty(dims[i+1], dims[i]) 
                                   for i in range(len(dims)-1)])
        self.encoder_bias = nn.ParameterList([torch.empty(dims[i+1]) for i in range(self.depth-1)])
        self.decoder_bias = nn.ParameterList([torch.empty(dims[i]) for i in range(self.depth-1)])
        ## -- ##
        self.__initweights__()
        self.activation = activation

    def __initweights__(self)->None:
        for weight, e_bias, d_bias in zip(self.weights, self.encoder_bias, self.decoder_bias):
            torch.nn.init.xavier_normal_(weight)
            torch.nn.init.zeros_(e_bias)
            torch.nn.init.zeros_(d_bias)
    
    def __encoder__(self, x:torch.Tensor)->torch.Tensor:
        for i, (weight, bias) in enumerate(zip(self.weights, self.encoder_bias)):
            x = F.linear(x, weight, bias)
            if i < self.depth -1:
                x = self.activation(x)
        return x
    def __decoder__(self, x:torch.Tensor)->torch.Tensor:
        reversed_weights = reversed(self.weights)
        reversed_bias = reversed(self.decoder_bias)
        for i, (weight, bias) in enumerate(zip(reversed_weights, reversed_bias)):
            if i < self.depth -1:
                x = self.activation(x)
            x = F.linear(x, weight.T, bias)
        return x

    def forward(self, 
                x:torch.Tensor, 
                encoder_output:bool = False)->torch.Tensor:
        enc_output = self.__encoder__(x)
        if encoder_output:
            return enc_output
        return self.__decoder__(enc_output)

## This dude is a bit sloppy...
class auto_encoder(nn.Module):
    def __init__(self, 
                 dims:list[int], 
                 activation:Callable = nn.GELU("tanh"),
                 **kwargs)->None:
        super().__init__(**kwargs)
        
        self.dims = dims
        self.depth = len(dims)
        self.activation = activation
        self.encoder = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1]) for i in range(self.depth-1)
        ])
        self.decoder = nn.ModuleList([
            nn.Linear(dims[i+1], dims[i]) for i in reversed(range(self.depth-1))
        ])

    def forward(self, x:torch.Tensor, encoder_output = False):
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < self.depth -1:
                x = self.activation(x)
        
        if encoder_output:
            return x

        for i, layer in enumerate(self.decoder):
            if i < self.depth -1:
                x = self.activation(x)
            x = layer(x)
        return x


if __name__== "__main__":
    pass