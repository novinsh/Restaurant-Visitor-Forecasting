import csv
import os


wd = os.getcwd()

data = []
with open(wd + '/air_visit_with_month.csv') as f:
    csvReader = csv.reader(f)
    for row in csvReader:
        date = row[1]
        year = date.split('-')[0]
        row.append(year)
        data.append(row)


data[0] = ['air_store_id', 'date', 'holiday', 'visitors', 'weekday', 'month', 'year']

with open(wd + '/air_visit_weekay_month_year.csv', 'w') as g:
    writer = csv.writer(g, delimiter=',')
    for row in data:
        writer.writerow(row)