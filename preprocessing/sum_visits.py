import os
import csv

wd = os.getcwd()

dates = {}
with open(wd + '/air_visit_with_weekday.csv') as f:
    csvReader = csv.reader(f)
    for row in csvReader:
        try:
            if row[1] in dates:
                dates[row[1]][1] += int(row[3])
            else:
                dates[row[1]] = [row[2], int(row[3]), row[4]]
        except:
            continue

dates_list = []
for k in dates:
    row = [k, dates[k][0], dates[k][1], dates[k][2]]
    dates_list.append(row)


dates_list.insert(0, ['date', 'holiday', 'visitors', 'weekday'])

for i in range(10):
    print(dates_list[i])

with open(wd + '/air_visit_sum.csv', 'w') as g:
    writer = csv.writer(g, delimiter=',')
    for row in dates_list:
        writer.writerow(row)