import torch
from torch import nn as nn
import torch.nn.functional as F
import types, typing
from typing import Callable

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

            
