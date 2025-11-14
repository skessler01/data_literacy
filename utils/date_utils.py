# Takes a year and a week number and returns the dates of monday and sunday of this calender week
from datetime import timedelta, date

def calender_week_to_dates(year: int, week: int):
    # If 'week' is given, return Monday–Sunday of that ISO week.
    first_day = date.fromisocalendar(year, week, 1)
    last_day = first_day + timedelta(days=6)

    return str(first_day), str(last_day)

def year_start_end(year: int):
    first_day = date(year, 1, 1)
    last_day = date(year, 12, 31)
    return str(first_day), str(last_day)