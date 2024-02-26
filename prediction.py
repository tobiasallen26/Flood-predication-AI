import torch
from torch import nn, tensor
from torch.utils.data import DataLoader, Dataset
from read_data import read_past_river_data, read_rain_data
import matplotlib.pyplot as plt
import numpy as np


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
        """print(river_data_avg[i:i+14])
        print(river_data_min[i:i+14])
        print(river_data_max[i:i+14])
        print(rain_data[i:i+14])
        print()
        
        print(river_data_avg[i+15])
        print(river_data_min[i+15])
        print(river_data_max[i+15])
        print()"""
        
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
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(59, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def train(dataloader, model, loss_fn, optimizer):
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
    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
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


if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    print(model)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    train_dataLoader, test_dataloader, all_data = arrange_data()
    
    test(test_dataloader, model, loss_fn)
    
    for i in range(1000):
        train(train_dataLoader, model, loss_fn, optimizer)
    
    test(test_dataloader, model, loss_fn)
    
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
        
    a = 0
    
    """rain_data = np.array(read_past_rain_data())
    rain_data /= max(rain_data)
    plt.plot(rain_data[-a:], label="rain data")"""

    plt.plot(avg_levels[-a:], label="avg levels")
    plt.plot(min_levels[-a:], label="min levels")
    plt.plot(max_levels[-a:], label="max levels")
    
    plt.plot(predicted_avg_levels[-a:], label="predicted avg levels")
    plt.plot(predicted_min_levels[-a:], label="predicted min levels")
    plt.plot(predicted_max_levels[-a:], label="predicted max levels")
    
    plt.legend()
    plt.show()
    