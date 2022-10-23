import torch
import torch.nn as nn
import numpy as np


class MultiLayerNN(nn.Module):

    def __init__(self, weights, bias_vecs, name='GenericNN'):
        super(MultiLayerNN, self).__init__()
        self.name = name
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()
        self.init_weights(weights, bias_vecs)
        self.in_dim = self.layers[0].weight.data.shape[1]

    def __str__(self):
        in_dim = self.layers[0].weight.data.shape[1]
        out_dim = self.layers[-1].weight.data.shape[0]
        s = self.name
        s += f"f:R^{in_dim} -> R^{out_dim} \n"
        for i, l in enumerate(self.layers):
            if isinstance(l, nn.Linear):
                s += f"W{int(i/2)}: {l.weight.data} \n"
                s += f"b{int(i/2)} : {l.bias.data} \n"
            if isinstance(l, nn.modules.activation.ReLU):
                s += f"ReLU\n"
        return s

    def verify_weight_dims(self, weights, bias_vecs):
        assert len(weights) == len(bias_vecs)
        for i, (w, b) in enumerate(zip(weights, bias_vecs)):
            assert w.shape[0] == b.shape[0]
            if i < len(weights) - 1:
                # not the last layer
                assert b.shape[0] == weights[i+1].shape[1]
            if i > 0:
                # after the first layer
                assert w.shape[1] == weights[i-1].shape[0]

    def init_weights(self, weights, bias_vecs):
        weights = [np.array(w) for w in weights]
        bias_vecs = [np.reshape(np.array(b), (len(b),)) for b in bias_vecs]
        self.verify_weight_dims(weights, bias_vecs)

        with torch.no_grad():
            for i, (w, b) in enumerate(zip(weights, bias_vecs)):
                l = nn.Linear(w.shape[1], w.shape[0])
                l.weight.copy_(torch.tensor(w))
                l.bias.copy_(torch.tensor(b))
                self.layers.append(l)
                self.layers.append(self.relu)
        return

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


def identity_map(dim, nlayers):
    weights   = []
    bias_vecs = []
    for l in range(nlayers):
        # build an identity dim x dim matrix
        weights.append(np.eye(dim))
        bias_vecs.append(np.zeros((dim, 1)))
    f = MultiLayerNN(weights, bias_vecs, name='Identity Map')
    return f


def null_map(in_dim, out_dim, nlayers = 3):
    # all but the last layer are fat
    weights   = []
    bias_vecs = []
    for l in range(nlayers):
        if l < nlayers - 1:
            weights.append(np.zeros((in_dim, in_dim)))
            bias_vecs.append(np.zeros((in_dim, 1)))
        else:
            # last layer
            weights.append(np.zeros((out_dim, in_dim)))
            bias_vecs.append(np.zeros((out_dim, 1)))
    f = MultiLayerNN(weights, bias_vecs, name='Null Map')
    return f


if __name__ == '__main__':
    f = identity_map(8, 2)
    f = null_map(4, 2)

    weights = [
        [[1, 0,0,0],
         [0, 1,0,0],
         [0, 0,1,0]],
        [[2, 0, 0],
         [0, 2, 0]],
        [[3, 0],
         [0, 3]]
    ]
    bias_vecs =[
        [1,1,1],
        [2,2],
        [3,3],
    ]
    f = MultiLayerNN(weights, bias_vecs)

    weights = [
        [[1, 0],
         [0, 1]],
        [[2, 0],
         [0, 2]],
        [[3, 0],
         [0, 3]]
    ]
    bias_vecs =[
        [1,1],
        [2,2],
        [3,3],
    ]
    f = MultiLayerNN(weights, bias_vecs)
