import numpy as np
import torch
import torch.nn as nn

class AbstractVerifier():

    def __init__(self, f=None):
        self.f = f
        self.relu = nn.ReLU()
        if f:
            self.nn_weights, self.nn_bias_vecs = self.get_weights_from_nn(self.f)

    def get_weights_from_nn(self, f):
        # only handles 'flat' ffnn's (for now)
        # https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
        weights, bias_vecs = [], []
        for i, l in enumerate(f.layers):
            if isinstance(l, nn.modules.linear.Linear):
                weights.append(l.weight.data.numpy())
                bias_vecs.append(l.bias.data.numpy())
            else:
                assert isinstance(l, nn.modules.activation.ReLU)
        return weights, bias_vecs

    def __str__(self):
        s = ''
        if not self.f:
            s += "No nn provided."
        else:
            s += f"f:R^{self.nn_weights[0].shape[1]} -> R^{self.nn_weights[-1].shape[0]}"
            s += f"\t{len(self.nn_weights)} layers"
        return s

