import os
import csv

wd = os.getcwd()

store_info = {}
with open(wd + '/air_store_info.csv') as f:
    csvReader = csv.reader(f)
    for row in csvReader:
        store_info[row[0]] = row[1:]

reserve_info = []
with open(wd + '/air_reserve.csv') as f:
    csvReader = csv.reader(f)
    for row in csvReader:
        reserve_info.append(row)

i = 1
reserve_info_conc = []
while i < len(reserve_info):
    try:
        id = reserve_info[i][0]
        visit_date = reserve_info[i][1].split()[0]
        visitors = int(reserve_info[i][3])
        j = 1
        while id == reserve_info[i+j][0] and visit_date == reserve_info[i+j][1].split()[0]:
            visitors += int(reserve_info[i+j][3])
            j += 1
        i += j
        row = [id, visit_date, visitors]
        reserve_info_conc.append(row)
    except Exception as e:
        print(e)
        break



visit_data = []
with open(wd + '/air_visit_with_weekday.csv') as f:
    csvReader = csv.reader(f)
    for row in csvReader:
        reserve_visitors = -1
        for row2 in reserve_info_conc:
            if row[0] == row2[0] and row[1] == row2[1]:
                reserve_visitors = row2[2]
        row = row + store_info[row[0]] + [reserve_visitors]
        visit_data.append(row)


visit_data[0] = ['air_store_id', 'date', 'holiday', 'visitors', 'weekday', 'air_genre_name', 'air_area_name', 'latitude', 'longitude', 'reserve_visitors']

with open(wd + '/air_visit_with_info.csv', 'w') as g:
    writer = csv.writer(g, delimiter=',')
    for row in visit_data:
        writer.writerow(row)


