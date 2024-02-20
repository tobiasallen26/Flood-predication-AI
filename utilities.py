import datetime

def dates(start, end, deliminator="_"):
    for day in range(int((end - start).days)+1):
        yield str(start + datetime.timedelta(day)).replace("-", deliminator)
