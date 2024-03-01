import requests
from datetime import datetime, timedelta, date
import os
from utilities import dates, get_last_date, update_last_date
from constants import most_recent_date, FIRST_RIVER_DATE
from passwords import meteomatics_username, meteomatics_password

RIVER_LEVEL_URL = r"https://riverlevels.uk/jesus-lock-sluice-auto-cambridge-cambridgeshire/data/csv"
RAIN_URL_FRONT = r"https://www.cl.cam.ac.uk/weather/data/daily-text/"
FUTURE_RAIN_URL_FRONT = r"https://api.meteomatics.com/"

headers = {"User-Agent": "Program to predict river water levels"}

def get_river_data():
    last_date = get_last_date()
    if str(most_recent_date) == last_date:
        print("most recent data is already stored")
        return
    update_last_date(str(datetime.now().date()))
    r = requests.get(RIVER_LEVEL_URL, headers=headers)
    # print(r.text)
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
        

def get_future_rain_data(use_temp_store=False):
    if use_temp_store:
        with open("rain_forcast.txt", "r") as file:
            data = file.read()
    else:
        time = datetime.now().strftime("%Y-%m-%dT00:00:00Z")
        time2 = (datetime.now() + timedelta(2)).strftime("%Y-%m-%dT00:00:00Z")
        URL = "https://" + meteomatics_username + ":" + meteomatics_password + "@api.meteomatics.com/"  + time + "--" + time2 + ":P1D/precip_24h:mm/52.2044000,0.113534/csv"
        print(URL)
        print(requests.get("https://" + meteomatics_username + ":" + meteomatics_password + "@api.meteomatics.com/"))
        data = requests.get(URL).text
        with open("rain_forcast.txt", "w") as file:
            data = file.write(data)
            
    d = []
    for i in data.split("\n")[1:-1]:
        d.append(float(i.split(";")[1]))
    return d


        
def get_data():
    get_river_data() 
    get_rain_data()

if __name__ == "__main__":
    get_data()
    get_future_rain_data(use_temp_store=True)
    