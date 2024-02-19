import requests

RIVER_LEVEL_URL = r"https://riverlevels.uk/jesus-lock-sluice-auto-cambridge-cambridgeshire/data/csv"

def get_data():
    r = requests.get(RIVΕR_LEVEL_URL)
