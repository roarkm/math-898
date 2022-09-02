# https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class FFNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out



model = FFNet(2, 2, 2).to(device)
print(model)

a = np.genfromtxt('data/a_bivariate_gaussian.csv',
                  delimiter=',')
b = np.genfromtxt('data/b_bivariate_gaussian.csv',
                  delimiter=',')
a = np.insert(a, 2, np.ones(len(a)), axis=1)
b = np.insert(b, 2, np.zeros(len(b)), axis=1)
ab = np.concatenate((a,b),axis=0)

# y_one_hot = np.zeros((ab.size, 2))
# y_one_hot[np.arange(ab[:,2].size), ab[:,2].astype('int')] = 1

t_x = torch.Tensor(ab[:,0:2])
t_y = torch.Tensor(ab[:,2].astype('int64'))
ds = TensorDataset(t_x, t_y.type(torch.uint8))
train_dataloader = DataLoader(ds, batch_size=len(ab))

# for X, y in train_dataloader:
        # print(f"Shape of X [N, C, H, W]: {X.shape}")
        # print(f"Shape of y: {y.shape} {y.dtype}")
        # break
# exit()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
