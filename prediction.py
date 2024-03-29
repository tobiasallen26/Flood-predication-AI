import torch
from torch import nn, tensor
from torch.utils.data import DataLoader, Dataset
from read_data import read_past_river_data, read_rain_data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from constants import *


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

batch_size = 64
test_percentage = 10

class CustomDataset(Dataset):
    def __init__(self, xy_list, transform=None, target_transform=None):
        self.inputs, self.outputs = zip(*xy_list)
        self.xy_list = xy_list
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        inp = self.inputs[idx]
        out = self.outputs[idx]
        if self.transform:
            inp = self.transform(inp)
        if self.target_transform:
            out = self.target_transform(out)
        # return inp, out
        return self.xy_list[idx]

def arrange_data():
    rain_data = read_rain_data()
    river_data = read_past_river_data()
    river_data_avg = river_data["avg_levels"]
    river_data_min = river_data["min_levels"]
    river_data_max = river_data["max_levels"]
    training_data = []
    for i in range(len(river_data_avg)-15):
        
        training_data.append([
            torch.tensor(list(tuple(river_data_avg[i:i+14])
            + tuple(river_data_min[i:i+14])
            + tuple(river_data_max[i:i+14])
            + tuple(rain_data[i:i+17])))
            ,torch.tensor([
                river_data_avg[i+15], 
                river_data_min[i+15], 
                river_data_max[i+15]])
            ])
        #print(training_data[-1])
        
    print(f"data set is {len(training_data)} items")
    training_data = CustomDataset(training_data)
    
    train_test_index = round(len(training_data)*(1-test_percentage/100))
    train_dl = DataLoader(training_data[:train_test_index], batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(training_data[train_test_index:], batch_size=int(batch_size/2))
    all_data_dl = DataLoader(training_data)
    return train_dl, test_dl, all_data_dl
    

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_elu_stack = nn.Sequential(
            nn.Linear(59, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 3),
        )
        
    def forward(self, x):
        if len(x.shape) != 1:
            x = self.flatten(x)
        logits = self.linear_elu_stack(x)
        return logits
    
def train(dataloader, model, loss_fn, optimizer, epoch=0):
    size = len(dataloader.dataset)
    model.train(True)
    for batch, (X, y) in enumerate(dataloader):
        # print(batch, X, y)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    loss, current = loss.item(), (batch + 1) * len(X)
    print(f"epoch: {epoch}  loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            
def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # print(pred, y)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss


if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    model = NeuralNetwork()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print(model)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.5)
    
    train_dataLoader, test_dataloader, all_data = arrange_data()
    
    losses = []
    losses.append(test(test_dataloader, model, loss_fn))
    
    plt.ion()
    
    for i in range(3000):
        train(train_dataLoader, model, loss_fn, optimizer, epoch=i)
        losses.append(test(test_dataloader, model, loss_fn))
        
        if i == 2:
            losses = []
        plt.clf()
        plt.plot(losses)
        plt.draw()
        plt.show()
        plt.pause(0.001)
        print(i)
        
    plt.ioff()
    
    test(test_dataloader, model, loss_fn)
    
    torch.save(model.state_dict(), MODEL_PATH)
    
    avg_levels, min_levels, max_levels = [], [], []
    predicted_avg_levels, predicted_min_levels, predicted_max_levels = [], [], []
    for X, y in all_data:
        avg_levels.append(float(y[0][0]))
        min_levels.append(float(y[0][1]))
        max_levels.append(float(y[0][2]))
        
        pred = model(X)
        predicted_avg_levels.append(float(pred[0][0]))
        predicted_min_levels.append(float(pred[0][1]))
        predicted_max_levels.append(float(pred[0][2]))
        
    
    rain_data = read_rain_data()
    river_data = read_past_river_data()
    pred_pred_avg = list(river_data["avg_levels"][:14])
    pred_pred_min = list(river_data["min_levels"][:14])
    pred_pred_max = list(river_data["max_levels"][:14])
    
    for i in range(len(rain_data)-17):
        pred_pred = model(torch.tensor(list(tuple(pred_pred_avg[i:i+14])
            + tuple(pred_pred_min[i:i+14])
            + tuple(pred_pred_max[i:i+14])
            + tuple(rain_data[i:i+17]))))
        
        pred_pred_avg.append(float(pred_pred[0]))
        pred_pred_min.append(float(pred_pred[1]))
        pred_pred_max.append(float(pred_pred[2]))
        
    a = 0
    
    plt.clf()
    
    """rain_data = np.array(read_past_rain_data())
    rain_data /= max(rain_data)
    plt.plot(rain_data[-a:], label="rain data")"""

    plt.plot(avg_levels[-a:], label="avg levels")
    plt.plot(min_levels[-a:], label="min levels")
    plt.plot(max_levels[-a:], label="max levels")
    
    plt.plot(predicted_avg_levels[-a:], label="predicted avg levels")
    plt.plot(predicted_min_levels[-a:], label="predicted min levels")
    plt.plot(predicted_max_levels[-a:], label="predicted max levels")
    
    plt.plot(pred_pred_avg[-a:], label="pred pred avg levels")
    plt.plot(pred_pred_min[-a:], label="pred pred min levels")
    plt.plot(pred_pred_max[-a:], label="pred pred max levels")
    
    plt.legend()
    plt.show()
    