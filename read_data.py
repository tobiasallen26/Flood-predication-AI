from utilities import dates
from constants import *
from get_data import get_future_rain_data

def read_past_river_data():
    
    with open("past_data.txt", "r") as file:
        data = file.read()
    
    data = data.split("\n")
    titles, data = data[0], data[1:]
    
    
    unfiltered_data = data
    data = []
    expected_dates = dates(FIRST_RIVER_DATE, most_recent_date, deliminator="-")
    d = 0
    last_accurate = None
    for expected_date in expected_dates:
        temp_unfiltered_data = unfiltered_data[d].split(",")
        # print(expected_date, unfiltered_data[d][0])
        
        if temp_unfiltered_data[0] == str(expected_date):
            unfiltered_data[d] = unfiltered_data[d].split(",")
            unfiltered_data[d][1] = float(unfiltered_data[d][1])
            unfiltered_data[d][2] = float(unfiltered_data[d][2])
            unfiltered_data[d][3] = float(unfiltered_data[d][3])
            data.append(unfiltered_data[d])
            last_accurate = unfiltered_data[d]
            d+=1
        else:
            print("missing river value for", expected_date)
            ## THIS COULD BE IMPROVED - by using linear interpolation
            if last_accurate:
                last_accurate[0] = str(expected_date)
                data.append(last_accurate)
            else:
                data.append([str(expected_date), 0.0, 0.0, 0.0])
            
    level_dates, min_levels, avg_levels, max_levels = zip(*data)
    print(len(data))
    
    return {"dates": level_dates, "min_levels": min_levels, "avg_levels": avg_levels, "max_levels": max_levels}

def read_past_rain_data():
    rain = []
    for d in dates(FIRST_RIVER_DATE, most_recent_date):
        with open(f".\past_weather_data\{d}.txt", "r") as file:
            data = file.read()
            
        data = data.split("\n")[:-1]
        
        try:
            r = float(data[-1].split("\t")[8])
        except IndexError:
            print("missing rain value for", d)
            # print(data)
            r = 0
        
        rain.append(r)
            
        # print(d, rain)
    print(len(rain))
        
    return rain

def read_rain_data():
    return read_past_rain_data() + get_future_rain_data(use_temp_store=True)
    
if __name__ == "__main__":
    read_past_river_data()
    read_rain_data()
    