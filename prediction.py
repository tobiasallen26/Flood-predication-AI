from neuralNetwork import neuralNetwork
from read_data import read_past_rain_data, read_past_river_data
import numpy as np

def input_filter(input):
    return np.transpose(input)

def comparison_array_func(ans):
    return np.transpose(ans)

def get_training_data():
    rain = read_past_rain_data()
    river_data = read_past_river_data()
    
    data = []
    for i in range(len(rain)-15):
        data.append(
            [rain[i:i+14] 
            + list(river_data["min_levels"][i:i+14]) 
            + list(river_data["avg_levels"][i:i+14]) 
            + list(river_data["max_levels"][i:i+14])
            , 
            [river_data["min_levels"][i+14],
            river_data["avg_levels"][i+14],
            river_data["max_levels"][i+14]
            ]])

    return data

if __name__ == "__main__":
    nn = neuralNetwork([56, 50, 50, 50, 50, 50, 3], input_filter, comparison_array_func)
    training_data = get_training_data()
    
    nn.train(training_data)
    nn.run(training_data[-1][0], training_data[-1][1])
    print(nn.get_output())
    