import matplotlib.pyplot as plt

from read_data import read_past_river_data, read_past_rain_data

def display_past_river_data():
    data = read_past_river_data()
    
    plt.plot(data["avg_levels"])
    plt.plot(data["max_levels"])
    plt.plot(data["min_levels"])
    plt.show()
    
def display_past_rain_data():
    data = read_past_rain_data()
    print(data)
    plt.plot(data)
    plt.show()
    
def display_all(n):
    rain_data = read_past_rain_data()
    river_data = read_past_river_data()
    
    fig, ax = plt.subplots(2)
    
    ax[0].plot(rain_data[-n:])
    
    ax[1].plot(river_data["avg_levels"][-n:])
    ax[1].plot(river_data["max_levels"][-n:])
    ax[1].plot(river_data["min_levels"][-n:])
    
    plt.show()

if __name__ == "__main__":
    display_all(30)
