import csv
import os

wd = os.getcwd()


store_inf = {}
with open(wd + '/air_store_info.csv') as d:
    csvReader = csv.reader(d)
    for row in csvReader:
        store_inf[row[0]] = row[1]


reserve_info = []
with open(wd + '/air_reserve_include_holidays.csv') as f:
    csvReader = csv.reader(f)
    for row in csvReader:
        id = row[0]
        row.append(store_inf[id])
        reserve_info.append(row)


with open(wd + '/air_reserve_incl_genre.csv', 'w') as g:
    writer = csv.writer(g, delimiter=',')
    for row in reserve_info:
        writer.writerow(row)
