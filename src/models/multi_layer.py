import torch
import torch.nn as nn
import numpy as np


class MultiLayerNN(nn.Module):


    def __init__(self, weights, bias_vecs):
        super(MultiLayerNN, self).__init__()
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()
        self.init_weights(weights, bias_vecs)
        self.in_dim = self.layers[0].weight.data.shape[1]


    def __str__(self):
        s = ""
        for i, l in enumerate(self.layers):
            if isinstance(l, nn.Linear):
                s += f"W{int(i/2)}: {l.weight.data} \n"
                s += f"b{int(i/2)} : {l.bias.data} \n"
            if isinstance(l, nn.modules.activation.ReLU):
                s += f"ReLU\n"
        return s


    def verify_weight_dims(self, weights, bias_vecs):
        assert len(weights) == len(bias_vecs)
        for i in range(0, len(weights)):
            _out_dim = weights[i].shape[0]
            if i < len(weights) - 1:
                assert _out_dim == len(bias_vecs[i+1])
            if i > 0:
                assert weights[i].shape[1] == weights[i-1].shape[0]
        return None


    def list_to_np(self, lmats):
        _m = []
        for m in lmats:
            _m.append(np.array(m))
        return _m


    def init_weights(self, weights, bias_vecs):
        weights = self.list_to_np(weights)
        self.verify_weight_dims(weights, bias_vecs)

        with torch.no_grad():
            for i, w in enumerate(weights):
                l = nn.Linear(w.shape[1], w.shape[0])
                l.weight.copy_(torch.tensor(w))
                l.bias.copy_(torch.tensor(bias_vecs[i]))
                self.layers.append(l)
                if i != len(weights)-1:
                    self.layers.append(self.relu)
        return


    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x



if __name__ == '__main__':
    weights = [
        [[1, 0],
         [0, 1]],
        [[2, 0],
         [0, 2]],
        [[3, 0],
         [0, 3]]
    ]
    bias_vecs =[
        (1,1),
        (2,2),
        (3,3),
    ]
    f = MultiLayerNN(weights, bias_vecs)
