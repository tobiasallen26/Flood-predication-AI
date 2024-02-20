import requests
from read_data import get_last_date
from datetime import datetime, timedelta, date
import os
from utilities import dates
from constants import most_recent_date, FIRST_RIVER_DATE

RIVER_LEVEL_URL = r"https://riverlevels.uk/jesus-lock-sluice-auto-cambridge-cambridgeshire/data/csv"
RAIN_URL_FRONT = r"https://www.cl.cam.ac.uk/weather/data/daily-text/"

headers = {"User-Agent": "Program to predict river water levels"}

def get_river_data():
    last_date = get_last_date()
    if str(most_recent_date) == last_date:
        print("most recent data is already stored")
        return
    r = requests.get(RIVER_LEVEL_URL, headers=headers)
    print(r.text)
    with open("past_data.txt", "w") as file:
        file.write(r.text)
    print("collected data")

    
def get_rain_data():
    for d in dates(FIRST_RIVER_DATE, most_recent_date):
        if os.path.exists(f".\past_weather_data\{d}.txt"):
            continue
        print(RAIN_URL_FRONT + d)
        r = requests.get(RAIN_URL_FRONT + d)
        print(d)
        with open(f".\past_weather_data\{d}.txt", "x") as file:
            file.write(r.text)
        
        
def get_data():
    get_river_data() 
    get_rain_data()

if __name__ == "__main__":
    get_data()
    