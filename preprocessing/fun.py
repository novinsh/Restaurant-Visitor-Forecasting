import csv
import os

wd = os.getcwd()


area_names = []
with open(wd + '/air_store_info.csv') as f:
    csvReader = csv.reader(f)
    for row in csvReader:
        area_names.append(row[2])

out = []
for name in list(set(area_names)):
    out.append('```{r}\nplot_this(\'%s\')\n```\n' % name)


with open(wd + '/names.txt', 'w') as g:
    for i in out[1:]:
        g.write(i)