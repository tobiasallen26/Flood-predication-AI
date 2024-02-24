import torch
from torch import nn
from read_data import read_past_river_data, read_past_rain_data


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def arrange_data():
    rain_data = read_past_rain_data()
    river_data = read_past_river_data()
    river_data_avg = river_data["avg_levels"]
    river_data_min = river_data["min_levels"]
    river_data_max = river_data["max_levels"]
    for i in range(len(rain_data)-15):
        print(river_data_avg[i:i+14])
        print(river_data_min[i:i+14])
        print(river_data_max[i:i+14])
        print(rain_data[i:i+14])
        print()
        print(river_data_avg[i+15])
        print(river_data_min[i+15])
        print(river_data_max[i+15])
        print()
    

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    print(model)
    
    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")
    
    arrange_data()
    