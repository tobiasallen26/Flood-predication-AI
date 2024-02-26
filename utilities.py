import datetime

def dates(start, end, deliminator="_"):
    for day in range(int((end - start).days)+1):
        yield str(start + datetime.timedelta(day)).replace("-", deliminator)
        
def get_last_date():
    with open("last_date.txt", "r") as file:
        return file.read()

def update_last_date(date):
    with open("last_date.txt", "w") as file:
        file.write(date)
