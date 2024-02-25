import torch
from torch import nn, tensor
from torch.utils.data import DataLoader, Dataset
from read_data import read_past_river_data, read_past_rain_data


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

batch_size = 64
test_percentage = 5

class CustomDataset(Dataset):
    def __init__(self, xy_list, transform=None, target_transform=None):
        self.inputs, self.outputs = zip(*xy_list)
        self.outputs = list(self.outputs)
        self.inputs = list(self.inputs)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        inp = tensor(self.inputs[idx])
        out = tensor(self.outputs[idx])
        if self.transform:
            inp = self.transform(inp)
        if self.target_transform:
            out = self.target_transform(out)
        return inp, out

def arrange_data():
    rain_data = read_past_rain_data()
    river_data = read_past_river_data()
    river_data_avg = river_data["avg_levels"]
    river_data_min = river_data["min_levels"]
    river_data_max = river_data["max_levels"]
    training_data = []
    for i in range(len(rain_data)-15):
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
            list(tuple(river_data_avg[i:i+14])
            + tuple(river_data_min[i:i+14])
            + tuple(river_data_max[i:i+14])
            + tuple(rain_data[i:i+14]))
            ,[
                river_data_avg[i+15], 
                river_data_min[i+15], 
                river_data_min[i+15]]
            ])
        # print(training_data[-1])
        
    print(f"data set is {len(training_data)} items")
    training_data = CustomDataset(training_data)
    
    train_test_index = round(len(training_data)*(1-test_percentage/100))
    train_dl = DataLoader(training_data[:train_test_index], batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(training_data[train_test_index:], batch_size=batch_size, shuffle=True)
    return train_dl, test_dl
    

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(56, 512),
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
    model.train()
    for a in dataloader:
        print(a)
    for batch, (X, y) in enumerate(dataloader):
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
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    print(model)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    train_dataLoader, test_dataloader = arrange_data()
    
    train(train_dataLoader, model, loss_fn, optimizer)
    