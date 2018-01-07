import csv
import os
import datetime

wd = os.getcwd()


holidays = {}
with open(wd + '/date_info.csv') as f:
    csvReader = csv.reader(f)
    for row in csvReader:
        holidays[row[0]] = row[2]



def day(date_string):

    def remove_zero(num):
        if num.startswith('0'):
            num = num[1]
        return int(num)

    year = remove_zero(date_string.split('-')[0])
    month = remove_zero(date_string.split('-')[1])
    day  = remove_zero(date_string.split('-')[2])

    day_num = datetime.date(year, month, day).weekday()
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return weekdays[day_num]





data = []
with open(wd + '/air_visit_data.csv') as f:
    csvReader = csv.reader(f)
    for row in csvReader:
        try:
            visit_date = row[1].split()[0]
            day_of_week = day(visit_date)
            if visit_date in holidays:
                new_col = holidays[visit_date]
                row = row[:2] + [new_col] + row[2:] + [day_of_week]
            data.append(row)
        except Exception as e:
            #print(e)
            continue



data.insert(0, ['air_store_id', 'date', 'holiday', 'visitors', 'weekday'])

with open(wd + '/air_visit_with_weekday.csv', 'w') as g:
    writer = csv.writer(g, delimiter=',')
    for row in data:
        writer.writerow(row)
