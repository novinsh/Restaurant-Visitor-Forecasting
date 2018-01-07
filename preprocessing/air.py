import csv
import os

wd = os.getcwd()


#holidays = []
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
        try:
            visit_date = row[1].split()[0]
            reserve_date = row[2].split()[0]
            if visit_date in holidays:
                new_col = holidays[visit_date]
                new_col2 = holidays[reserve_date]
                row = row[:2] + [new_col] + row[2:3] + [new_col2] + row[3:]
            data.append(row)
        except Exception as e:
            #print(e)
            continue


#for i in range(10):
#    print(data[i])
#for k in holidays:
#    print(k, holidays[k])

data[0] = ['air_store_id', 'visit_datetime', 'visit_date_holiday', 'reserve_datetime', 'reserve_date_holiday', 'reserve_visitors']

with open(wd + '/air_reserve_include_holidays.csv', 'w') as g:
    writer = csv.writer(g, delimiter=',')
    for row in data:
        writer.writerow(row)







