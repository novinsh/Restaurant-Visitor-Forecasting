import csv
import os

wd = os.getcwd()





holidays = {}
with open(wd + '/date_info.csv') as f:
    csvReader = csv.reader(f)
    for row in csvReader:
        #holidays.append(row)
        holidays[row[0]] = row[2]


data = []
with open(wd + '/air_reserve.csv') as f:
    csvReader = csv.reader(f)
    for row in csvReader:
        data.append(row)


data = data[1:]
#for i in range(10):
#    print(data[i][1].split()[0])

print(len(data))


data_fixed = []
i = 1
while i < len(data):
    try:
        visit_date = data[i][1].split()[0]
        visitors = int(data[i][3])
        j = 1
        while visit_date == data[i+j][1].split()[0]:
            visitors += int(data[j][3])
            j += 1
        row = data[i][:1] + [visit_date, holidays[visit_date], j, visitors]
        data_fixed.append(row)
        i+=j
    except Exception as e:
        row = data[i][:2] + [holidays[visit_date], j, visitors]
        print('i: ', i)
        print('j: ', j)
        print(e)
        print()
        break



data_fixed.insert(0, ['air_store_id', 'visit_date', 'visit_date_holiday', 'number_of_reservations', 'reserve_visitors'])

for i in range(10):
    print(data_fixed[i])

with open(wd + '/air_reserve_concatenated.csv', 'w') as g:
    writer = csv.writer(g, delimiter=',')
    for row in data_fixed:
        writer.writerow(row)